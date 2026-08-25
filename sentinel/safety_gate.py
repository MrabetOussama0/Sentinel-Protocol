"""
sentinel.safety_gate
====================
Universal pre-execution safety check (is_action_safe), legacy Earth-command
validator (validate_command), and the comms-blackout survival loop.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

from sentinel.decision_engine import THREAT_CONSERVATISM


# ---------------------------------------------------------------------------
# Action semantic groups
# ---------------------------------------------------------------------------

_ADVANCE_CMDS = frozenset({
    "move_forward", "continue_heading", "increase_speed",
    "resume_traverse", "proceed", "advance",
})
_ANTENNA_CMDS = frozenset({
    "deploy_antenna", "raise_antenna", "extend_mast",
    "open_solar_panel", "deploy_instrument",
})
_HIGH_POWER_CMDS = frozenset({
    "deploy_antenna", "raise_antenna", "transmit_data",
    "queue_transmission", "activate_drill", "run_diagnostics",
    "enable_heaters", "full_sensor_sweep",
})
_MOVEMENT_CMDS = frozenset({
    "move_forward", "continue_heading", "increase_speed",
    "resume_traverse", "proceed", "advance",
    "move_backward", "reverse", "turn_left", "turn_right",
    "change_heading", "reposition",
    # rover-native corrective actions
    "reverse_to_safe_distance", "navigate_to_sunlight",
    "reposition_to_safety", "small_reverse",
})
_COMMS_CMDS = frozenset({
    "transmit_data", "queue_transmission", "send_telemetry",
    "uplink_report", "broadcast_status",
})
_ALWAYS_SAFE = frozenset({
    "stop", "hold", "hold_in_place", "hold_position",
    "emergency_full_stop", "cut_motors",
})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class SafetyCheckResult(NamedTuple):
    safe:       bool
    action:     str
    reason:     str   # empty if safe
    blocked_by: str   # threat type that caused the block, or empty


class ValidationResult(NamedTuple):
    verdict:      str   # 'APPROVED' or 'BLOCKED'
    command:      str
    reason:       str   # empty if approved
    earth_report: str   # watsonx block report (empty if approved)


class SurvivalStep(NamedTuple):
    phase:        str             # 'HOLD'|'REPOSITION'|'WAIT'|'BATTERY_RESCUE'|'ESCALATE'
    proposed:     str
    safety:       SafetyCheckResult
    executed:     bool
    sensor_state: dict
    note:         str


# ---------------------------------------------------------------------------
# Universal safety gate
# ---------------------------------------------------------------------------

def is_action_safe(
    proposed_action: str,
    sensor_state:    dict,
    active_threats:  list[str],
    comm_delay_s:    float = 780,
) -> SafetyCheckResult:
    """Universal pre-execution safety check for ANY rover action.

    Applies to both Earth-sent commands and rover-generated autonomous actions.

    Parameters
    ----------
    proposed_action : String action identifier.
    sensor_state    : Current raw sensor readings dict.
    active_threats  : List of currently active threat type strings.
    comm_delay_s    : One-way comm delay in seconds.

    Returns
    -------
    SafetyCheckResult — (safe, action, reason, blocked_by)
    """
    action = proposed_action.strip().lower()

    if action in _ALWAYS_SAFE:
        return SafetyCheckResult(safe=True, action=proposed_action, reason="", blocked_by="")

    for threat in active_threats:

        if threat == "cliff_edge" and action in _ADVANCE_CMDS:
            dist  = sensor_state.get("distance_m", float("inf"))
            speed = sensor_state.get("drift_speed_ms", 0.0)
            if speed > 0:
                tth_adj = (dist / speed) * THREAT_CONSERVATISM["cliff_edge"]
                rtt     = comm_delay_s * 2
                if tth_adj <= rtt:
                    return SafetyCheckResult(
                        safe=False, action=proposed_action,
                        reason=(f"cliff edge {dist:.1f} m ahead; "
                                f"adj TTH {tth_adj:.0f} s \u2264 RTT {rtt:.0f} s"),
                        blocked_by="cliff_edge",
                    )

        if threat == "rockfall" and action in _MOVEMENT_CMDS:
            dist  = sensor_state.get("debris_dist_m", float("inf"))
            speed = sensor_state.get("debris_speed_ms", 0.0)
            if speed > 0 and (dist / speed) <= 30.0:
                eta = dist / speed
                return SafetyCheckResult(
                    safe=False, action=proposed_action,
                    reason=f"debris {dist:.1f} m at {speed:.1f} m/s \u2014 impact ETA {eta:.1f} s",
                    blocked_by="rockfall",
                )

        if threat == "dust_storm" and action in _ANTENNA_CMDS:
            wind  = sensor_state.get("wind_speed_ms", 0.0)
            opdep = sensor_state.get("optical_depth", 0.0)
            if wind >= 15.0:
                return SafetyCheckResult(
                    safe=False, action=proposed_action,
                    reason=f"wind {wind:.1f} m/s \u2265 15 m/s structural limit",
                    blocked_by="dust_storm",
                )
            if opdep >= 0.6:
                return SafetyCheckResult(
                    safe=False, action=proposed_action,
                    reason=f"optical depth {opdep:.3f} \u2265 0.60 \u2014 particulate risk",
                    blocked_by="dust_storm",
                )

        if threat == "battery_critical" and action in _HIGH_POWER_CMDS:
            charge = sensor_state.get("charge_pct", 100.0)
            if charge <= 10.0:
                return SafetyCheckResult(
                    safe=False, action=proposed_action,
                    reason=f"battery {charge:.1f}% \u2264 10% \u2014 high-power action risks shutdown",
                    blocked_by="battery_critical",
                )

        if threat == "comms_blackout" and action in _COMMS_CMDS:
            elev = sensor_state.get("relay_elevation_deg", 90.0)
            if elev <= 8.0:
                return SafetyCheckResult(
                    safe=False, action=proposed_action,
                    reason=f"relay at {elev:.1f}\u00b0 (cutoff 8\u00b0) \u2014 transmission would fail",
                    blocked_by="comms_blackout",
                )

    return SafetyCheckResult(safe=True, action=proposed_action, reason="", blocked_by="")


# ---------------------------------------------------------------------------
# Legacy Earth-command validator  (wraps is_action_safe for single-threat use)
# ---------------------------------------------------------------------------

def validate_command(
    command:      str,
    sensor_state: dict,
    threat_type:  str | None = None,
    comm_delay_s: float = 780,
    _block_report_fn=None,
) -> ValidationResult:
    """Validate an incoming Earth command against the rover's current sensor state.

    Parameters
    ----------
    command         : Incoming command string from Earth.
    sensor_state    : Current sensor readings dict.
    threat_type     : Active threat category, or None.
    comm_delay_s    : One-way comm delay in seconds.
    _block_report_fn: Optional callable(command, threat_type, sensors, reason) → str
                      for generating a watsonx Earth report on blocks.
                      If None, earth_report is an empty string.
    """
    if not threat_type:
        return ValidationResult(verdict="APPROVED", command=command, reason="", earth_report="")

    result = is_action_safe(command, sensor_state, [threat_type], comm_delay_s)

    if result.safe:
        return ValidationResult(verdict="APPROVED", command=command, reason="", earth_report="")

    earth_report = ""
    if _block_report_fn is not None:
        earth_report = _block_report_fn(command, threat_type, sensor_state, result.reason)

    return ValidationResult(
        verdict="BLOCKED",
        command=command,
        reason=result.reason,
        earth_report=earth_report,
    )


# ---------------------------------------------------------------------------
# Comms-blackout survival loop
# ---------------------------------------------------------------------------

def blackout_survival_loop(
    sensor_state:          dict,
    comm_delay_s:          float = 780,
    battery_rescue_thresh: float = 12.0,
    max_wait_steps:        int   = 6,
) -> Iterator[SurvivalStep]:
    """Autonomous survival generator for comms-blackout situations.

    Every proposed action is validated with is_action_safe() before execution.

    Phases: HOLD → REPOSITION → WAIT → BATTERY_RESCUE → (ESCALATE if blocked)
    """
    state   = dict(sensor_state)
    threats = ["comms_blackout"]

    # HOLD
    hold_check = is_action_safe("hold_in_place", state, threats, comm_delay_s)
    yield SurvivalStep(
        phase="HOLD", proposed="hold_in_place", safety=hold_check,
        executed=hold_check.safe, sensor_state=dict(state),
        note="Blackout detected \u2014 attempting immediate stop.",
    )

    # Detect stacked cliff hazard
    dist  = state.get("distance_m", float("inf"))
    speed = state.get("drift_speed_ms", 0.0)
    if speed > 0:
        tth_adj = (dist / speed) * THREAT_CONSERVATISM.get("cliff_edge", 0.8)
        if tth_adj <= comm_delay_s * 2:
            threats.append("cliff_edge")

    # REPOSITION
    fwd_check = is_action_safe("move_forward", state, threats, comm_delay_s)
    if not fwd_check.safe:
        rev_check = is_action_safe("reverse_to_safe_distance", state, threats, comm_delay_s)
        if rev_check.safe:
            state["distance_m"]    = state.get("distance_m", 0.0) + 50.0
            state["drift_speed_ms"] = 0.0
            threats = [t for t in threats if t != "cliff_edge"]
            yield SurvivalStep(
                phase="REPOSITION", proposed="reverse_to_safe_distance",
                safety=rev_check, executed=True, sensor_state=dict(state),
                note="Forward path blocked (cliff). Reversing to safe distance.",
            )
        else:
            stop_check = is_action_safe("emergency_full_stop", state, threats, comm_delay_s)
            yield SurvivalStep(
                phase="ESCALATE", proposed="emergency_full_stop",
                safety=stop_check, executed=True, sensor_state=dict(state),
                note="Corrective reverse blocked. Executing emergency full stop.",
            )
    else:
        yield SurvivalStep(
            phase="REPOSITION", proposed="hold_in_place",
            safety=hold_check, executed=True, sensor_state=dict(state),
            note="Forward path clear \u2014 no reposition needed, maintaining hold.",
        )

    # WAIT / BATTERY_RESCUE
    for wait_i in range(max_wait_steps):
        state["charge_pct"] = max(0.0, state.get("charge_pct", 100.0) - 0.8)
        charge = state["charge_pct"]

        if charge <= battery_rescue_thresh:
            nav_check = is_action_safe("navigate_to_sunlight", state, threats, comm_delay_s)
            if nav_check.safe:
                state["charge_pct"] = min(100.0, charge + 30.0)
                yield SurvivalStep(
                    phase="BATTERY_RESCUE", proposed="navigate_to_sunlight",
                    safety=nav_check, executed=True, sensor_state=dict(state),
                    note=(f"Charge at {charge:.1f}% \u2264 {battery_rescue_thresh}% threshold. "
                          f"Navigating to sunlit charging location."),
                )
            else:
                yield SurvivalStep(
                    phase="BATTERY_RESCUE", proposed="navigate_to_sunlight",
                    safety=nav_check, executed=False, sensor_state=dict(state),
                    note=f"Charge critical but navigation blocked ({nav_check.reason}).",
                )
                stop_check = is_action_safe("emergency_full_stop", state, threats, comm_delay_s)
                yield SurvivalStep(
                    phase="ESCALATE", proposed="emergency_full_stop",
                    safety=stop_check, executed=True, sensor_state=dict(state),
                    note="Battery critical and movement blocked \u2014 emergency full stop.",
                )
            break
        else:
            wait_check = is_action_safe("hold_in_place", state, threats, comm_delay_s)
            yield SurvivalStep(
                phase="WAIT", proposed="hold_in_place",
                safety=wait_check, executed=True, sensor_state=dict(state),
                note=(f"Waiting for Earth contact. Charge {charge:.1f}%. "
                      f"Re-check {wait_i + 1}/{max_wait_steps}."),
            )
