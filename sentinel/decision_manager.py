"""
sentinel.decision_manager
=========================
Central arbitrator that resolves conflicts between autonomous hazard response,
Earth commands, and emergency safety actions.

This module closes the architecture gaps identified in the review:

  M1 — Earth response timeout / GREEN→YELLOW→RED escalation from waiting
  M2 — Anomaly detection wired into every tick
  M3 — Sensor monitoring continues during Earth-command execution;
        in-flight commands are interrupted if a new threat emerges
  M5 — Single authority that enforces priority: EMERGENCY > AUTONOMOUS > EARTH

Priority order (highest wins)
──────────────────────────────
  1. EMERGENCY  — RED-tier threat detected; act immediately regardless of
                  any active Earth command.
  2. AUTONOMOUS — YELLOW-tier: execute holding action, notify Earth.
                  GREEN-tier + no active Earth command: wait for Earth.
  3. EARTH      — Execute an approved Earth command, but re-validate it
                  against fresh sensor data on every tick while it is active.

Public API
──────────
  MissionState         — complete snapshot of rover state for one tick
  DecisionManager      — stateful tick processor; call .tick(sensors) each cycle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from sentinel.decision_engine import (
    DecisionTier,
    THREAT_CONSERVATISM,
    classify_threat,
)
from sentinel.simulator import choose_holding_action
from sentinel.safety_gate import (
    SafetyCheckResult,
    ValidationResult,
    is_action_safe,
    validate_command,
)
from sentinel.anomaly import SensorWindow, classify_sensor_pattern, AnomalyResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class EarthCommand(NamedTuple):
    """A command received from Earth, queued for execution."""
    command:      str
    received_tick: int


@dataclass
class MissionState:
    """Complete output of one decision-manager tick.

    Attributes
    ----------
    tick              : Tick counter (0-based).
    sensors           : Raw sensor readings dict for this tick.
    active_threats    : List of currently active threat type strings.
    time_to_harm_s    : Best TTH estimate for the primary threat (or inf).
    tier              : Final decided tier after all arbitration.
    anomaly           : AnomalyResult from the IsolationForest (or None).
    holding_action    : YELLOW-tier holding action, or None.
    earth_cmd         : The Earth command being considered this tick, or None.
    cmd_result        : ValidationResult for earth_cmd, or None.
    cmd_interrupted   : True if an in-flight Earth command was interrupted.
    earth_timeout     : True if an Earth response timeout caused tier escalation.
    ticks_waiting     : How many ticks the rover has been waiting for Earth.
    priority_source   : Which authority determined the final tier:
                        'EMERGENCY' | 'AUTONOMOUS' | 'EARTH' | 'IDLE'
    notes             : List of human-readable event notes for this tick.
    """
    tick:            int
    sensors:         dict
    active_threats:  list[str]
    time_to_harm_s:  float
    tier:            DecisionTier
    anomaly:         AnomalyResult | None
    holding_action:  str | None
    earth_cmd:       EarthCommand | None
    cmd_result:      ValidationResult | None
    cmd_interrupted: bool
    earth_timeout:   bool
    ticks_waiting:   int
    priority_source: str
    notes:           list[str]


# ---------------------------------------------------------------------------
# DecisionManager
# ---------------------------------------------------------------------------

class DecisionManager:
    """Stateful tick processor that implements the full rover decision loop.

    Parameters
    ----------
    comm_delay_s         : One-way comm delay to Earth in seconds.
    tick_duration_s      : Wall-clock duration of one tick in seconds.
    earth_timeout_ticks  : After this many waiting ticks with no Earth response,
                           force a tier re-evaluation (GREEN may become YELLOW/RED
                           because sensors have worsened). Default: derived from
                           RTT ÷ tick_duration so one round-trip = one timeout.
    anomaly_window_size  : Number of sensor readings in the rolling anomaly window.
    use_anomaly_model    : Whether to run classify_sensor_pattern() each tick.
    """

    def __init__(
        self,
        comm_delay_s:        float = 780,
        tick_duration_s:     float = 30,
        earth_timeout_ticks: int | None = None,
        anomaly_window_size: int = 10,
        use_anomaly_model:   bool = True,
    ):
        self.comm_delay_s    = comm_delay_s
        self.tick_duration_s = tick_duration_s
        # Default: one full RTT worth of ticks
        rtt_ticks = max(1, int((comm_delay_s * 2) / tick_duration_s))
        self.earth_timeout_ticks = earth_timeout_ticks if earth_timeout_ticks is not None else rtt_ticks
        self.use_anomaly_model   = use_anomaly_model

        self._tick_count:         int             = 0
        self._window:             SensorWindow    = SensorWindow(_size=anomaly_window_size)
        self._earth_cmd_queue:    list[EarthCommand] = []   # commands waiting to be processed
        self._active_cmd:         EarthCommand | None = None   # command currently executing
        self._ticks_waiting:      int             = 0   # ticks spent waiting for Earth response
        self._green_since_tick:   int | None      = None   # tick when we entered GREEN waiting

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def queue_earth_command(self, command: str) -> None:
        """Enqueue a command arriving from Earth for validation on the next tick."""
        self._earth_cmd_queue.append(EarthCommand(command=command, received_tick=self._tick_count))

    def tick(self, sensors: dict) -> MissionState:
        """Process one sensor reading and return the full mission state.

        This is the single entry point for the control loop.  Call it once per
        tick with the latest sensor readings.

        Steps
        -----
        1. Push sensors into the anomaly window.
        2. Run anomaly detection (if enabled).
        3. Determine active threats (from scenario + anomaly result).
        4. Classify tier using classify_threat().
        5. Check Earth timeout → force re-evaluation if waiting too long.
        6. Arbitrate priority: EMERGENCY overrides EARTH; EARTH validated live.
        7. If an Earth command is active, re-validate against fresh sensors —
           interrupt if now unsafe.
        8. Process any newly queued Earth commands.
        9. Emit MissionState.
        """
        notes: list[str] = []
        tick = self._tick_count
        self._tick_count += 1

        # ── Step 1: push to anomaly window ───────────────────────────────────
        self._window.push(sensors)

        # ── Step 2: anomaly detection ─────────────────────────────────────────
        anomaly: AnomalyResult | None = None
        if self.use_anomaly_model:
            try:
                anomaly = classify_sensor_pattern(
                    sensors, self._window, comm_delay_s=self.comm_delay_s
                )
            except Exception as exc:
                notes.append(f"anomaly model error: {exc}")

        # ── Step 3: determine active threats ──────────────────────────────────
        # Primary threat comes from the scenario being simulated (caller sets
        # this via the `scenario_threat` context, injected through sensors keys
        # matching known threat signatures).  Anomaly layer may add more.
        primary_threat: str | None = None
        if anomaly and anomaly.is_anomaly and anomaly.threat_type:
            primary_threat = anomaly.threat_type

        active_threats: list[str] = [primary_threat] if primary_threat else []

        # ── Step 4: classify tier from best TTH estimate ──────────────────────
        tth    = float("inf")
        tier   = DecisionTier.GREEN

        if primary_threat and primary_threat in THREAT_CONSERVATISM:
            # Use anomaly's computed tier (which now calls classify_threat internally)
            tier = anomaly.tier if anomaly else DecisionTier.GREEN
            # Also compute raw TTH for reporting
            from sentinel.anomaly import _estimate_tth
            raw_tth = _estimate_tth(primary_threat, sensors)
            if raw_tth is not None:
                tth = raw_tth
        elif primary_threat == "unclassified_anomaly":
            tier = DecisionTier.YELLOW
            notes.append("unclassified anomaly: holding and notifying Earth")

        # ── Step 5: Earth timeout escalation (M1) ─────────────────────────────
        earth_timeout = False
        if tier == DecisionTier.GREEN:
            # We are in GREEN — safe to wait.  Track how long we've been waiting.
            if self._green_since_tick is None:
                self._green_since_tick = tick
            self._ticks_waiting = tick - self._green_since_tick
            if self._ticks_waiting >= self.earth_timeout_ticks and self._ticks_waiting > 0:
                # Earth hasn't responded within one RTT worth of ticks.
                # Re-evaluate: sensors may have worsened.
                # Nudge: treat remaining margin as if conservatism is tighter.
                if active_threats and primary_threat:
                    timeout_tier = classify_threat(
                        primary_threat,
                        tth * 0.85,   # 15% extra tightening for elapsed wait time
                        self.comm_delay_s,
                    )
                    if timeout_tier != DecisionTier.GREEN:
                        tier = timeout_tier
                        earth_timeout = True
                        notes.append(
                            f"Earth timeout after {self._ticks_waiting} ticks "
                            f"({self._ticks_waiting * self.tick_duration_s:.0f}s waiting) "
                            f"— tier escalated to {tier.value}"
                        )
        else:
            self._green_since_tick = None
            self._ticks_waiting    = 0

        # ── Step 6: priority arbitration ──────────────────────────────────────
        holding_action:  str | None = None
        priority_source: str        = "IDLE"
        cmd_interrupted              = False
        cmd_result: ValidationResult | None = None
        earth_cmd:  EarthCommand | None = None

        if tier == DecisionTier.RED:
            # EMERGENCY — autonomous immediate action, interrupt anything
            priority_source = "EMERGENCY"
            if self._active_cmd is not None:
                notes.append(
                    f"RED tier: interrupting active Earth command "
                    f"'{self._active_cmd.command}'"
                )
                cmd_interrupted = True
                earth_cmd       = self._active_cmd
                self._active_cmd = None
            self._green_since_tick = None
            self._ticks_waiting    = 0

        elif tier == DecisionTier.YELLOW:
            # AUTONOMOUS holding action
            priority_source = "AUTONOMOUS"
            if active_threats:
                holding_action = choose_holding_action(
                    active_threats[0], sensors, self.comm_delay_s
                )
            # Still interrupt unsafe Earth commands
            if self._active_cmd is not None:
                recheck = is_action_safe(
                    self._active_cmd.command, sensors, active_threats, self.comm_delay_s
                )
                if not recheck.safe:
                    notes.append(
                        f"YELLOW tier: active command '{self._active_cmd.command}' "
                        f"now unsafe ({recheck.reason}) — interrupting"
                    )
                    cmd_interrupted = True
                    earth_cmd       = self._active_cmd
                    self._active_cmd = None

        else:
            # GREEN — Earth command can proceed or be started
            priority_source = "EARTH" if self._active_cmd else "IDLE"

            # ── Step 7: re-validate active Earth command with fresh sensors ───
            if self._active_cmd is not None:
                earth_cmd = self._active_cmd
                recheck   = is_action_safe(
                    self._active_cmd.command, sensors, active_threats, self.comm_delay_s
                )
                cmd_result = ValidationResult(
                    verdict      = "APPROVED" if recheck.safe else "BLOCKED",
                    command      = self._active_cmd.command,
                    reason       = recheck.reason,
                    earth_report = "",
                )
                if not recheck.safe:
                    notes.append(
                        f"In-flight command '{self._active_cmd.command}' interrupted: "
                        f"{recheck.reason}"
                    )
                    cmd_interrupted  = True
                    self._active_cmd = None
                else:
                    notes.append(f"Command '{self._active_cmd.command}' continuing — sensors still safe")
                    # Mark command as completed after one tick (simplification for demo;
                    # real systems would track multi-tick progress separately)
                    self._active_cmd = None

        # ── Step 8: process newly queued Earth commands ───────────────────────
        if self._earth_cmd_queue:
            next_cmd = self._earth_cmd_queue.pop(0)
            if tier == DecisionTier.RED:
                # Never start a new Earth command during RED
                r = ValidationResult(
                    verdict="BLOCKED", command=next_cmd.command,
                    reason="RED-tier emergency active — Earth command rejected",
                    earth_report="",
                )
                notes.append(f"Rejected queued command '{next_cmd.command}' — RED tier active")
                earth_cmd  = next_cmd
                cmd_result = r
            else:
                # Validate and potentially start the command
                r = validate_command(
                    command      = next_cmd.command,
                    sensor_state = sensors,
                    threat_type  = active_threats if active_threats else None,
                    comm_delay_s = self.comm_delay_s,
                )
                earth_cmd  = next_cmd
                cmd_result = r
                if r.verdict == "APPROVED":
                    self._active_cmd = next_cmd
                    priority_source  = "EARTH"
                    notes.append(f"Earth command '{next_cmd.command}' approved and started")
                else:
                    notes.append(
                        f"Earth command '{next_cmd.command}' BLOCKED: {r.reason}"
                    )

        return MissionState(
            tick            = tick,
            sensors         = sensors,
            active_threats  = active_threats,
            time_to_harm_s  = round(tth, 1) if tth != float("inf") else tth,
            tier            = tier,
            anomaly         = anomaly,
            holding_action  = holding_action,
            earth_cmd       = earth_cmd,
            cmd_result      = cmd_result,
            cmd_interrupted = cmd_interrupted,
            earth_timeout   = earth_timeout,
            ticks_waiting   = self._ticks_waiting,
            priority_source = priority_source,
            notes           = notes,
        )

    def reset(self) -> None:
        """Reset all state (new scenario run)."""
        self._tick_count       = 0
        self._window           = SensorWindow(_size=self._window._size)
        self._earth_cmd_queue  = []
        self._active_cmd       = None
        self._ticks_waiting    = 0
        self._green_since_tick = None
