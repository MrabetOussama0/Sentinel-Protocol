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
    wind_ms, dust_gcm3, wind_ramp, dust_ramp, tick_dur_s = 4.0, 0.001, 2.5, 0.004, 60
    for _ in range(ticks):
        optical_depth = dust_gcm3 * 100 * (wind_ms ** 0.3)
        next_od = (dust_gcm3 + dust_ramp) * 100 * ((wind_ms + wind_ramp) ** 0.3)
        od_rate = max((next_od - optical_depth) / tick_dur_s, 1e-9)
        tth     = max((1.0 - optical_depth) / od_rate, 0.0)
        yield (
            {"wind_speed_ms": round(wind_ms, 2),
             "dust_density_gcm3": round(dust_gcm3, 4),
             "optical_depth": round(optical_depth, 4)},
            tth,
        )
        wind_ms   += wind_ramp
        dust_gcm3 += dust_ramp


def _battery_critical_model(ticks: int):
    charge_pct, draw, draw_accel, tick_dur_s = 18.0, 0.8, 0.12, 60
    for _ in range(ticks):
        tth = (charge_pct / draw) * tick_dur_s
        yield ({"charge_pct": round(charge_pct, 2), "draw_pct_per_tick": round(draw, 3)}, tth)
        charge_pct = max(0.0, charge_pct - draw)
        draw      += draw_accel


def _rockfall_model(ticks: int):
    seismic_g, debris_dist_m, debris_speed = 0.05, 80.0, 1.5
    seismic_ramp, speed_ramp, tick_dur_s   = 0.08, 2.0, 5
    for _ in range(ticks):
        tth = debris_dist_m / debris_speed if debris_speed > 0 else float("inf")
        yield (
            {"seismic_g": round(seismic_g, 3),
             "debris_dist_m": round(debris_dist_m, 2),
             "debris_speed_ms": round(debris_speed, 2)},
            tth,
        )
        debris_dist_m = max(0.0, debris_dist_m - debris_speed * tick_dur_s)
        debris_speed += speed_ramp
        seismic_g    += seismic_ramp


def _comms_blackout_model(ticks: int):
    elevation_deg, descent_rate, tick_dur_s = 42.0, 1.8, 60
    for _ in range(ticks):
        remaining_deg = max(elevation_deg - 5.0, 0.0)
        eff_rate      = descent_rate * (1 + 0.04 * (42.0 - elevation_deg))
        tth           = max((remaining_deg / eff_rate) * tick_dur_s, 0.0)
        yield (
            {"relay_elevation_deg": round(elevation_deg, 2),
             "effective_descent_rate": round(eff_rate, 3)},
            tth,
        )
        elevation_deg = max(0.0, elevation_deg - descent_rate)


_MODELS: dict[str, object] = {
    "cliff_edge":       _cliff_edge_model,
    "dust_storm":       _dust_storm_model,
    "battery_critical": _battery_critical_model,
    "rockfall":         _rockfall_model,
    "comms_blackout":   _comms_blackout_model,
}


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
    """
    if threat_type not in _MODELS:
        raise ValueError(f"Unknown threat type: {threat_type!r}")
    model = _MODELS[threat_type](ticks)
    for tick_idx, (sensors, tth) in enumerate(model):
        tier = classify_threat(threat_type, tth, comm_delay_s)
        ha   = (
            choose_holding_action(threat_type, sensors, comm_delay_s)
            if tier == DecisionTier.YELLOW
            else None
        )
        yield TickState(
            tick=tick_idx,
            sensors=sensors,
            time_to_harm_s=round(tth, 1),
            tier=tier,
            holding_action=ha,
        )
