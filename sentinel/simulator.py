"""
sentinel.simulator
==================
Tick-by-tick physics models for each threat type, choose_holding_action(),
TickState, and run_scenario().
"""

from __future__ import annotations

import math
from typing import Iterator, NamedTuple

from sentinel.decision_engine import (
    DecisionTier,
    ThreatType,
    THREAT_CONSERVATISM,
    classify_threat,
)


# ---------------------------------------------------------------------------
# Tick state
# ---------------------------------------------------------------------------

class TickState(NamedTuple):
    tick:           int
    sensors:        dict        # raw physics readings
    time_to_harm_s: float       # derived estimate fed into classify_threat
    tier:           DecisionTier
    holding_action: str | None  # 'hold_in_place' | 'reposition_to_safety' | None
    robot_state:    str         # one-word physical state: MOVING | HOLDING | STOPPED | REPOSITIONING | CHARGING | SURVIVAL
    robot_activity: str         # plain-language sentence: what is the rover doing RIGHT NOW


# ---------------------------------------------------------------------------
# YELLOW-tier holding action selector
# ---------------------------------------------------------------------------

_REPOSITION_UNSAFE_WIND_MS = 20.0   # m/s — too dangerous to traverse
_REPOSITION_UNSAFE_CHARGE  =  5.0   # %   — insufficient power to reposition


def choose_holding_action(
    threat_type:  str,
    sensor_state: dict,
    comm_delay_s: float = 780,
) -> str:
    """Return 'hold_in_place' or 'reposition_to_safety' for a YELLOW-tier tick.

    Rules
    -----
    cliff_edge       → hold_in_place         (any movement near a cliff is unsafe)
    dust_storm       → reposition_to_safety  (unless wind ≥ 20 m/s)
    battery_critical → reposition_to_safety  (unless charge ≤ 5 %)
    rockfall         → hold_in_place         (movement increases debris exposure)
    comms_blackout   → hold_in_place         (navigating blind is unsafe)
    """
    if threat_type == "cliff_edge":
        return "hold_in_place"
    if threat_type == "dust_storm":
        return (
            "hold_in_place"
            if sensor_state.get("wind_speed_ms", 0.0) >= _REPOSITION_UNSAFE_WIND_MS
            else "reposition_to_safety"
        )
    if threat_type == "battery_critical":
        return (
            "hold_in_place"
            if sensor_state.get("charge_pct", 100.0) <= _REPOSITION_UNSAFE_CHARGE
            else "reposition_to_safety"
        )
    if threat_type in ("rockfall", "comms_blackout"):
        return "hold_in_place"
    return "hold_in_place"


# ---------------------------------------------------------------------------
# Robot state/activity label builder
# ---------------------------------------------------------------------------

def _build_robot_status(
    threat_type:    str,
    tier:           DecisionTier,
    holding_action: str | None,
    sensors:        dict,
    tick:           int,
) -> tuple[str, str]:
    """Return (robot_state, robot_activity) for the current tick.

    robot_state   : Short uppercase word shown in the status badge.
    robot_activity: Human-readable sentence shown in the status card.
    """
    tv = tier.value

    # ── GREEN — nominal operations ──────────────────────────────────────────
    if tv == "GREEN":
        if threat_type == "cliff_edge":
            dist  = sensors.get("distance_m", "?")
            speed = sensors.get("drift_speed_ms", 0.0)
            return ("MOVING",
                    f"Traversing nominal path. Cliff detected {dist} m ahead — "
                    f"drift {speed:.4f} m/s. Monitoring only; awaiting Earth guidance.")

        if threat_type == "dust_storm":
            wind = sensors.get("wind_speed_ms", 0.0)
            od   = sensors.get("optical_depth", 0.0)
            return ("MOVING",
                    f"Continuing science traverse. Light dust detected "
                    f"(wind {wind:.2f} m/s, optical depth {od:.4f}). "
                    f"No action needed — Earth notified.")

        if threat_type == "battery_critical":
            charge = sensors.get("charge_pct", 0.0)
            draw   = sensors.get("draw_pct_per_tick", 0.0)
            return ("MOVING",
                    f"Operating normally. Battery at {charge:.1f}% "
                    f"(draw {draw:.3f}%/tick). Monitoring discharge trend.")

        if threat_type == "rockfall":
            dist  = sensors.get("debris_dist_m", "?")
            speed = sensors.get("debris_speed_ms", 0.0)
            return ("MOVING",
                    f"Normal traverse. Seismic event detected — debris {dist} m "
                    f"at {speed:.3f} m/s. Monitoring; Earth alerted.")

        if threat_type == "comms_blackout":
            elev  = sensors.get("relay_elevation_deg", 0.0)
            rate  = sensors.get("effective_descent_rate", 0.0)
            return ("MOVING",
                    f"Full contact with Earth. Relay at {elev:.1f}° "
                    f"(descending {rate:.3f}°/tick). "
                    f"Diagnostics and science ops nominal.")

        if threat_type in ("full_blackout", "blackout_survival"):
            elev = sensors.get("relay_elevation_deg", 0.0)
            return ("MOVING",
                    f"Relay link active at {elev:.1f}°. Nominal traversal. "
                    f"Pre-blackout systems check running.")

        return ("MOVING", "Nominal operations. No immediate action required.")

    # ── YELLOW — holding / repositioning ───────────────────────────────────
    if tv == "YELLOW":
        if holding_action == "hold_in_place":
            if threat_type == "cliff_edge":
                dist = sensors.get("distance_m", "?")
                return ("HOLDING",
                        f"⚠️ Brakes applied. Holding position — cliff {dist} m ahead. "
                        f"Earth notified; waiting for response within budget window.")
            if threat_type == "rockfall":
                dist = sensors.get("debris_dist_m", "?")
                return ("HOLDING",
                        f"⚠️ Motors stopped. Debris {dist} m and closing — "
                        f"movement would increase exposure. Waiting for Earth.")
            if threat_type == "comms_blackout":
                elev = sensors.get("relay_elevation_deg", 0.0)
                return ("HOLDING",
                        f"⚠️ Relay descending ({elev:.1f}°). Holding position — "
                        f"navigating without contact is unsafe. Earth notified.")
            if threat_type in ("full_blackout", "blackout_survival"):
                elev = sensors.get("relay_elevation_deg", 0.0)
                return ("HOLDING",
                        f"⚠️ Relay at {elev:.1f}° — approaching horizon. "
                        f"Halting traverse. Conserving energy. Pre-blackout checklist active.")
            return ("HOLDING",
                    f"⚠️ Holding in place. Hazard detected — awaiting Earth response.")

        if holding_action == "reposition_to_safety":
            if threat_type == "dust_storm":
                wind = sensors.get("wind_speed_ms", 0.0)
                od   = sensors.get("optical_depth", 0.0)
                return ("REPOSITIONING",
                        f"⚠️ Storm intensifying (wind {wind:.2f} m/s, OD {od:.4f}). "
                        f"Repositioning toward shelter. Panels partially shielded.")
            if threat_type == "battery_critical":
                charge = sensors.get("charge_pct", 0.0)
                return ("REPOSITIONING",
                        f"⚠️ Battery at {charge:.1f}%. Navigating toward sunlit "
                        f"charging area. Non-essential systems suspended.")
            return ("REPOSITIONING",
                    "⚠️ Repositioning to safer location. Earth notified.")

        return ("HOLDING", "⚠️ Holding — hazard within response window. Earth notified.")

    # ── RED — autonomous action ─────────────────────────────────────────────
    if tv == "RED":
        if threat_type == "cliff_edge":
            dist = sensors.get("distance_m", "?")
            return ("STOPPED",
                    f"🚨 AUTONOMOUS ACTION. Cliff {dist} m — TTH < RTT. "
                    f"Emergency reverse engaged. Earth cannot respond in time. "
                    f"Notifying Earth after manoeuvre.")

        if threat_type == "dust_storm":
            wind = sensors.get("wind_speed_ms", 0.0)
            od   = sensors.get("optical_depth", 0.0)
            return ("STOPPED",
                    f"🚨 AUTONOMOUS ACTION. Storm critical (wind {wind:.2f} m/s, OD {od:.4f}). "
                    f"Solar panels shielded. Instruments stowed. "
                    f"Sheltering in place until storm passes.")

        if threat_type == "battery_critical":
            charge = sensors.get("charge_pct", 0.0)
            return ("CHARGING",
                    f"🚨 AUTONOMOUS ACTION. Battery {charge:.1f}% — critical threshold. "
                    f"Entering low-power survival mode. Non-essential systems off. "
                    f"Navigating to nearest charging position.")

        if threat_type == "rockfall":
            dist = sensors.get("debris_dist_m", "?")
            spd  = sensors.get("debris_speed_ms", 0.0)
            return ("STOPPED",
                    f"🚨 AUTONOMOUS ACTION. Debris {dist} m at {spd:.3f} m/s — "
                    f"impact imminent. Executing maximum emergency evasion. "
                    f"Earth notified after.")

        if threat_type == "comms_blackout":
            elev = sensors.get("relay_elevation_deg", 0.0)
            return ("SURVIVAL",
                    f"🚨 BLACKOUT. Relay at {elev:.1f}° — contact lost. "
                    f"Entering autonomous survival loop: holding position, "
                    f"conserving battery, monitoring for relay return.")

        if threat_type in ("full_blackout", "blackout_survival"):
            charge = sensors.get("charge_pct", sensors.get("battery_pct", 0.0))
            wind   = sensors.get("wind_speed_ms", 0.0)
            return ("SURVIVAL",
                    f"🚨 FULL BLACKOUT. No Earth contact. Survival mode active. "
                    f"Battery {charge:.1f}%, wind {wind:.2f} m/s. "
                    f"Self-managing: dust protection, battery triage, "
                    f"holding position until comms restored.")

        return ("STOPPED",
                "🚨 AUTONOMOUS ACTION. TTH exceeded threshold — acting now.")

    return ("MOVING", "Status unknown.")


# ---------------------------------------------------------------------------
# Physics models — one per threat type
# ---------------------------------------------------------------------------

def _cliff_edge_model(ticks: int):
    distance_m, speed_ms, accel, tick_dur_s = 100.0, 0.02, 0.003, 30
    for _ in range(ticks):
        tth = distance_m / speed_ms if speed_ms > 0 else float("inf")
        yield ({"distance_m": round(distance_m, 3), "drift_speed_ms": round(speed_ms, 4)}, tth)
        distance_m = max(0.0, distance_m - speed_ms * tick_dur_s)
        speed_ms  += accel


def _dust_storm_model(ticks: int):
    # Slow-build storm: starts as a light breeze with trace dust; optical depth
    # rises steadily over ~16 ticks before panels are critically obscured.
    # Panel shutdown threshold raised to 1.0 optical-depth units (unchanged) but
    # initial wind and dust are now much lower so the approach is gradual.
    # Produces: GREEN ticks 0-3, YELLOW ticks 4-15, RED tick 16+  (RTT=1560 s).
    wind_ms, dust_gcm3, wind_ramp, dust_ramp, tick_dur_s = 0.5, 0.00001, 0.5, 0.0001, 60
    for _ in range(ticks):
        optical_depth = dust_gcm3 * 100 * (wind_ms ** 0.3)
        next_od = (dust_gcm3 + dust_ramp) * 100 * ((wind_ms + wind_ramp) ** 0.3)
        od_rate = max((next_od - optical_depth) / tick_dur_s, 1e-9)
        tth     = max((1.0 - optical_depth) / od_rate, 0.0)
        yield (
            {"wind_speed_ms": round(wind_ms, 2),
             "dust_density_gcm3": round(dust_gcm3, 5),
             "optical_depth": round(optical_depth, 4)},
            tth,
        )
        wind_ms   += wind_ramp
        dust_gcm3 += dust_ramp


def _dust_storm_slow_model(ticks: int):
    """Very gentle dust storm that rises to YELLOW but never hits RED within ticks.

    Used for the 'dust_storm_yellow_hold' story — demonstrates that tier
    progression is NOT always GREEN → YELLOW → RED.
    Builds slowly: GREEN ticks 0-5, YELLOW ticks 6+, never RED within 25 ticks.
    comm_delay_s=120 (short delay) so TTH stays above RTT even when severe.
    """
    wind_ms, dust_gcm3, wind_ramp, dust_ramp, tick_dur_s = 0.2, 0.000005, 0.22, 0.00004, 60
    for _ in range(ticks):
        optical_depth = dust_gcm3 * 100 * (wind_ms ** 0.3)
        next_od = (dust_gcm3 + dust_ramp) * 100 * ((wind_ms + wind_ramp) ** 0.3)
        od_rate = max((next_od - optical_depth) / tick_dur_s, 1e-9)
        tth     = max((1.0 - optical_depth) / od_rate, 0.0)
        yield (
            {"wind_speed_ms": round(wind_ms, 2),
             "dust_density_gcm3": round(dust_gcm3, 5),
             "optical_depth": round(optical_depth, 4)},
            tth,
        )
        wind_ms   += wind_ramp
        dust_gcm3 += dust_ramp


def _battery_critical_model(ticks: int):
    # Slow drain scenario: battery starts at 30 %, draw rate begins at 0.3 %/tick
    # and accelerates gradually.  Provides a clean multi-tick GREEN → YELLOW → RED
    # progression rather than starting already in RED.
    # Produces: GREEN ticks 0-1, YELLOW ticks 2-6, RED tick 7+  (RTT=1560 s).
    charge_pct, draw, draw_accel, tick_dur_s = 30.0, 0.3, 0.05, 60
    for _ in range(ticks):
        tth = (charge_pct / draw) * tick_dur_s
        yield ({"charge_pct": round(charge_pct, 2), "draw_pct_per_tick": round(draw, 3)}, tth)
        charge_pct = max(0.0, charge_pct - draw)
        draw      += draw_accel


def _rockfall_model(ticks: int):
    # Distant seismic event: debris starts 2500 m away, rolling slowly.
    # Speed increases each tick as debris accelerates under gravity.
    # Produces: GREEN ticks 0-3, YELLOW ticks 4-8, RED tick 9+  (RTT=1560 s).
    seismic_g, debris_dist_m, debris_speed = 0.02, 2500.0, 0.2
    seismic_ramp, speed_ramp, tick_dur_s   = 0.02, 0.10, 10
    for _ in range(ticks):
        tth = debris_dist_m / debris_speed if debris_speed > 0 else float("inf")
        yield (
            {"seismic_g": round(seismic_g, 3),
             "debris_dist_m": round(debris_dist_m, 1),
             "debris_speed_ms": round(debris_speed, 3)},
            tth,
        )
        debris_dist_m = max(0.0, debris_dist_m - debris_speed * tick_dur_s)
        debris_speed += speed_ramp
        seismic_g    += seismic_ramp


def _comms_blackout_model(ticks: int):
    # Satellite starts high (80°), descends at 1.5°/tick with increasing rate
    # as orbital geometry compresses near the horizon.
    # Produces: GREEN ticks 0-3, YELLOW ticks 4-12, RED tick 13+  (RTT=1560 s).
    elevation_deg, descent_rate, tick_dur_s = 80.0, 1.5, 90
    for _ in range(ticks):
        remaining_deg = max(elevation_deg - 5.0, 0.0)
        eff_rate      = descent_rate * (1 + 0.06 * (80.0 - elevation_deg))
        tth           = max((remaining_deg / eff_rate) * tick_dur_s, 0.0)
        yield (
            {"relay_elevation_deg": round(elevation_deg, 2),
             "effective_descent_rate": round(eff_rate, 3)},
            tth,
        )
        elevation_deg = max(0.0, elevation_deg - descent_rate)


def _full_blackout_model(ticks: int):
    """Blackout scenario: comms are ALREADY lost from tick 0.

    The relay went below the horizon before the scenario starts.
    The rover must self-manage everything autonomously — dust rising,
    battery draining, cliff nearby — until a rescue relay window opens.

    Sensors: relay_elevation_deg (starts at 3°, continues falling then
    recovers after tick 14 simulating a second relay coming over the horizon),
    battery_pct (drains), wind_speed_ms (dust building), distance_m (cliff).

    No Earth contact the entire time. Demonstrates full autonomous survival.
    """
    # relay starts just below contact threshold (5°), drops further, then a
    # second relay rises starting at tick 14 (rescue window)
    relay_deg    = 3.0
    battery_pct  = 55.0
    wind_ms      = 1.0
    dist_m       = 180.0   # cliff in the distance
    drift_ms     = 0.005

    for tick_i in range(ticks):
        # relay geometry: descends 1.2°/tick until tick 8, then flat, then rises
        # when rescue relay appears at tick 14
        if tick_i < 8:
            relay_deg = max(0.0, relay_deg - 1.2)
        elif tick_i < 14:
            relay_deg = max(0.0, relay_deg - 0.2)
        else:
            # rescue relay rises 2°/tick
            relay_deg = min(90.0, relay_deg + 2.0)

        battery_pct = max(0.0, battery_pct - 0.6 - tick_i * 0.04)
        wind_ms     = min(25.0, wind_ms + 0.8)
        dist_m      = max(0.0, dist_m - drift_ms * 30)

        # TTH based on battery (primary concern during blackout)
        draw_rate = 0.6 + tick_i * 0.04
        tth = (battery_pct / draw_rate) * 60 if draw_rate > 0 else float("inf")

        yield (
            {
                "relay_elevation_deg": round(relay_deg, 2),
                "battery_pct":         round(battery_pct, 2),
                "wind_speed_ms":       round(wind_ms, 2),
                "distance_m":          round(dist_m, 2),
                "drift_speed_ms":      round(drift_ms, 4),
            },
            tth,
        )


_MODELS: dict[str, object] = {
    "cliff_edge":           _cliff_edge_model,
    "dust_storm":           _dust_storm_model,
    "dust_storm_slow":      _dust_storm_slow_model,
    "battery_critical":     _battery_critical_model,
    "rockfall":             _rockfall_model,
    "comms_blackout":       _comms_blackout_model,
    "full_blackout":        _full_blackout_model,
}

# Register full_blackout in THREAT_CONSERVATISM for classification
# (uses same conservatism as battery_critical — discharge is predictable)
THREAT_CONSERVATISM.setdefault("full_blackout", 0.90)
THREAT_CONSERVATISM.setdefault("blackout_survival", 0.90)
THREAT_CONSERVATISM.setdefault("dust_storm_slow", 0.90)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_scenario(
    threat_type:  str,
    ticks:        int   = 20,
    comm_delay_s: float = 780,
) -> Iterator[TickState]:
    """Tick-by-tick scenario simulator for a single threat type.

    Yields
    ------
    TickState
        tick           : Tick index (0-based).
        sensors        : Raw sensor readings dict for the current tick.
        time_to_harm_s : Derived time-to-harm estimate (seconds).
        tier           : DecisionTier result from classify_threat().
        holding_action : 'hold_in_place' | 'reposition_to_safety' | None.
                         Non-None only on YELLOW ticks.
        robot_state    : Uppercase word describing physical state (MOVING, HOLDING, etc.)
        robot_activity : Plain-language sentence describing the rover's current action.
    """
    # Map display threat types to THREAT_CONSERVATISM keys
    _classify_as = {
        "dust_storm_slow": "dust_storm",
        "full_blackout":   "comms_blackout",
    }

    if threat_type not in _MODELS:
        raise ValueError(f"Unknown threat type: {threat_type!r}")

    classify_key = _classify_as.get(threat_type, threat_type)
    model = _MODELS[threat_type](ticks)

    for tick_idx, (sensors, tth) in enumerate(model):
        tier = classify_threat(classify_key, tth, comm_delay_s)
        ha   = (
            choose_holding_action(classify_key, sensors, comm_delay_s)
            if tier == DecisionTier.YELLOW
            else None
        )
        robot_state, robot_activity = _build_robot_status(
            threat_type, tier, ha, sensors, tick_idx
        )
        yield TickState(
            tick=tick_idx,
            sensors=sensors,
            time_to_harm_s=round(tth, 1),
            tier=tier,
            holding_action=ha,
            robot_state=robot_state,
            robot_activity=robot_activity,
        )
