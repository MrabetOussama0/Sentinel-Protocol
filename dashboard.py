"""
Sentinel Protocol — Live Mission Dashboard
==========================================
Run with:
    streamlit run dashboard.py

Architecture note
-----------------
Every Streamlit run renders the FULL UI first, then schedules the next tick
via st.rerun() at the very bottom.  time.sleep() is called only after all
widgets are painted so the browser always shows a complete frame.
"""

import os
import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Literal, NamedTuple

import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Protocol",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Core engine — tiers, dataclass, classify_threat()
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTier(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"

@dataclass
class Threat:
    threat_type:    str
    time_to_harm_s: float
    comm_delay_s:   float

    @property
    def round_trip_s(self) -> float:
        return self.comm_delay_s * 2

    @property
    def time_margin_ratio(self) -> float:
        if self.round_trip_s == 0:
            return float("inf")
        return self.time_to_harm_s / self.round_trip_s

THREAT_CONSERVATISM: dict[str, float] = {
    "cliff_edge":       0.80,
    "dust_storm":       0.90,
    "battery_critical": 0.95,
    "rockfall":         0.70,
    "comms_blackout":   1.00,
}

def classify_threat(threat_type: str, time_to_harm_s: float, comm_delay_s: float) -> DecisionTier:
    conservatism = THREAT_CONSERVATISM[threat_type]
    threat = Threat(
        threat_type    = threat_type,
        time_to_harm_s = time_to_harm_s * conservatism,
        comm_delay_s   = comm_delay_s,
    )
    ratio = threat.time_margin_ratio
    if ratio > 2.0:
        return DecisionTier.GREEN
    elif ratio > 1.0:
        return DecisionTier.YELLOW
    else:
        return DecisionTier.RED

# ─────────────────────────────────────────────────────────────────────────────
# Scenario simulator
# ─────────────────────────────────────────────────────────────────────────────

class TickState(NamedTuple):
    tick:           int
    sensors:        dict
    time_to_harm_s: float
    tier:           DecisionTier
    holding_action: str | None  # 'hold_in_place' | 'reposition_to_safety' | None

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
        yield ({"wind_speed_ms": round(wind_ms, 2),
                "dust_density_gcm3": round(dust_gcm3, 4),
                "optical_depth": round(optical_depth, 4)}, tth)
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
        yield ({"seismic_g": round(seismic_g, 3),
                "debris_dist_m": round(debris_dist_m, 2),
                "debris_speed_ms": round(debris_speed, 2)}, tth)
        debris_dist_m = max(0.0, debris_dist_m - debris_speed * tick_dur_s)
        debris_speed += speed_ramp
        seismic_g    += seismic_ramp

def _comms_blackout_model(ticks: int):
    elevation_deg, descent_rate, tick_dur_s = 42.0, 1.8, 60
    for _ in range(ticks):
        remaining_deg = max(elevation_deg - 5.0, 0.0)
        eff_rate      = descent_rate * (1 + 0.04 * (42.0 - elevation_deg))
        tth           = max((remaining_deg / eff_rate) * tick_dur_s, 0.0)
        yield ({"relay_elevation_deg": round(elevation_deg, 2),
                "effective_descent_rate": round(eff_rate, 3)}, tth)
        elevation_deg = max(0.0, elevation_deg - descent_rate)

_MODELS = {
    "cliff_edge":       _cliff_edge_model,
    "dust_storm":       _dust_storm_model,
    "battery_critical": _battery_critical_model,
    "rockfall":         _rockfall_model,
    "comms_blackout":   _comms_blackout_model,
}

# Sensor thresholds used by choose_holding_action
_REPOSITION_UNSAFE_WIND_MS = 20.0
_REPOSITION_UNSAFE_CHARGE  =  5.0

def choose_holding_action(threat_type: str, sensor_state: dict,
                           comm_delay_s: float = 780) -> str:
    """Return 'hold_in_place' or 'reposition_to_safety' for a YELLOW-tier tick."""
    if threat_type == "cliff_edge":
        return "hold_in_place"
    if threat_type == "dust_storm":
        return "hold_in_place" if sensor_state.get("wind_speed_ms", 0) >= _REPOSITION_UNSAFE_WIND_MS else "reposition_to_safety"
    if threat_type == "battery_critical":
        return "hold_in_place" if sensor_state.get("charge_pct", 100) <= _REPOSITION_UNSAFE_CHARGE else "reposition_to_safety"
    if threat_type in ("rockfall", "comms_blackout"):
        return "hold_in_place"
    return "hold_in_place"

def run_scenario(threat_type: str, ticks: int, comm_delay_s: float) -> list[TickState]:
    """Pre-compute all ticks into a list (Streamlit-friendly — no generator in session state)."""
    results = []
    for i, (sensors, tth) in enumerate(_MODELS[threat_type](ticks)):
        tier = classify_threat(threat_type, tth, comm_delay_s)
        ha   = choose_holding_action(threat_type, sensors, comm_delay_s) if tier == DecisionTier.YELLOW else None
        results.append(TickState(tick=i, sensors=sensors,
                                 time_to_harm_s=round(tth, 1), tier=tier,
                                 holding_action=ha))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# watsonx.ai — generate_reasoning()
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

_WATSONX_READY = False
_wx_model      = None

def _init_watsonx() -> bool:
    global _wx_model, _WATSONX_READY
    if _WATSONX_READY:
        return True
    api_key    = os.environ.get("WATSONX_API_KEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    if not api_key or "your_ibm_cloud" in api_key:
        return False
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        _wx_model = ModelInference(
            model_id    = "ibm/granite-4-h-small",
            credentials = Credentials(url="https://eu-de.ml.cloud.ibm.com", api_key=api_key),
            project_id  = project_id,
        )
        _WATSONX_READY = True
        return True
    except Exception:
        return False

_SYSTEM_PROMPT = (
    "You are the autonomous reasoning system of a planetary rover named Sentinel. "
    "Write a single professional mission-log sentence (maximum 40 words) explaining "
    "the decision made. Be factual, precise, and terse — like a flight engineer "
    "writing a flight log entry. Output only the log sentence, nothing else."
)

_USER_TEMPLATE = (
    "SITUATION:\n"
    "  Threat      : {threat_type}\n"
    "  Sensor data : {sensors}\n"
    "  Time-to-harm: {time_to_harm_s:.1f} s\n"
    "  Round-trip comm delay: {round_trip_s:.0f} s\n"
    "  Adjusted ratio (TTH/RTT): {ratio:.3f}\n"
    "  Decision tier: {tier}\n"
    "  Required action: {action}\n"
    "\nWrite the mission log entry for this tick."
)

@st.cache_data(show_spinner=False)
def generate_reasoning(threat_type: str, sensors_repr: str, time_to_harm_s: float,
                       round_trip_s: float, ratio: float, tier_val: str, action: str) -> str:
    """Cached per unique (tick inputs) — API called once per tick, never on reruns."""
    if not _init_watsonx():
        return "(watsonx not configured — check .env credentials)"
    tick_data = {
        "threat_type":    threat_type,
        "sensors":        sensors_repr,
        "time_to_harm_s": time_to_harm_s,
        "round_trip_s":   round_trip_s,
        "ratio":          ratio,
        "tier":           tier_val,
        "action":         action,
    }
    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(**tick_data)},
        ]
        resp = _wx_model.chat(
            messages=messages,
            params={"max_tokens": 80, "temperature": 0.3, "repetition_penalty": 1.05},
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(model error: {e})"

# ─────────────────────────────────────────────────────────────────────────────
# Lookup tables — string keys only (enum-class-reload-safe)
# ─────────────────────────────────────────────────────────────────────────────

TIER_COLOR = {"GREEN": "#16a34a", "YELLOW": "#ca8a04", "RED": "#dc2626"}
TIER_BG    = {"GREEN": "#dcfce7", "YELLOW": "#fef9c3", "RED": "#fee2e2"}
TIER_ICON  = {"GREEN": "🟢",      "YELLOW": "🟡",      "RED": "🔴"}

TIER_ACTION = {
    "GREEN":  "Wait for Earth response.",
    "YELLOW": "Execute safe holding action; notify Earth immediately.",
    "RED":    "Act autonomously NOW; notify Earth after action.",
}

SCENARIO_LABELS = {
    "cliff_edge":       "🪨  Cliff Edge",
    "dust_storm":       "🌪️  Dust Storm",
    "battery_critical": "🔋  Battery Critical",
    "rockfall":         "⛰️  Rockfall",
    "comms_blackout":   "📡  Comms Blackout",
}

SENSOR_UNITS = {
    "distance_m":             "m",
    "drift_speed_ms":         "m/s",
    "wind_speed_ms":          "m/s",
    "dust_density_gcm3":      "g/cm³",
    "optical_depth":          "",
    "charge_pct":             "%",
    "draw_pct_per_tick":      "%/tick",
    "seismic_g":              "g",
    "debris_dist_m":          "m",
    "debris_speed_ms":        "m/s",
    "relay_elevation_deg":    "°",
    "effective_descent_rate": "°/tick",
}

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tier_badge_html(tier_val: str, holding_action: str | None = None) -> str:
    color = TIER_COLOR[tier_val]
    bg    = TIER_BG[tier_val]
    icon  = TIER_ICON[tier_val]
    ha_line = ""
    if holding_action:
        ha_icon = "🧲" if holding_action == "hold_in_place" else "🔄"
        ha_label = holding_action.replace("_", " ")
        ha_line = (
            f'<br><span style="color:{color};font-size:0.78rem;'
            f'background:rgba(0,0,0,0.06);border-radius:4px;padding:2px 8px;">'
            f'{ha_icon} {ha_label}</span>'
        )
    return (
        f'<div style="background:{bg};border:2px solid {color};border-radius:12px;'
        f'padding:18px 28px;text-align:center;">'
        f'<span style="font-size:2.4rem;font-weight:800;color:{color};">'
        f'{icon} {tier_val}</span><br>'
        f'<span style="color:{color};font-size:0.85rem;">{TIER_ACTION[tier_val]}</span>'
        f'{ha_line}'
        f'</div>'
    )

def _ratio_bar_html(ratio: float) -> str:
    pct      = min(ratio / 3.0, 1.0) * 100
    tier_val = "GREEN" if ratio > 2 else ("YELLOW" if ratio > 1 else "RED")
    color    = TIER_COLOR[tier_val]
    return (
        f'<div style="background:#e5e7eb;border-radius:6px;height:14px;width:100%;">'
        f'<div style="background:{color};width:{pct:.1f}%;height:14px;border-radius:6px;"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:0.7rem;color:#6b7280;margin-top:2px;">'
        f'<span>0 — RED</span><span>1× RTT</span><span>2× RTT</span><span>3×</span></div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Command validation — validate_command() + block-report generator
# ─────────────────────────────────────────────────────────────────────────────

_ADVANCE_CMDS  = frozenset({"move_forward", "continue_heading", "increase_speed",
                             "resume_traverse", "proceed", "advance"})
_ANTENNA_CMDS  = frozenset({"deploy_antenna", "raise_antenna", "extend_mast",
                             "open_solar_panel", "deploy_instrument"})
_HIGH_PWR_CMDS = frozenset({"deploy_antenna", "raise_antenna", "transmit_data",
                             "queue_transmission", "activate_drill", "run_diagnostics",
                             "enable_heaters", "full_sensor_sweep"})
_MOVEMENT_CMDS = frozenset({"move_forward", "continue_heading", "increase_speed",
                             "resume_traverse", "proceed", "advance",
                             "move_backward", "reverse", "turn_left", "turn_right",
                             "change_heading", "reposition"})
_COMMS_CMDS    = frozenset({"transmit_data", "queue_transmission", "send_telemetry",
                             "uplink_report", "broadcast_status"})

_BLOCK_SYSTEM = (
    "You are the autonomous safety system of a planetary rover named Sentinel. "
    "A command from Earth has been blocked because it conflicts with an active hazard. "
    "Write a single professional sentence (maximum 45 words) reporting the block back to Earth. "
    "Be factual and terse — like a flight engineer writing a status update. "
    "Output only the sentence, nothing else."
)
_BLOCK_USER = (
    "Blocked command : {command}\n"
    "Active threat   : {threat_type}\n"
    "Sensor state    : {sensors}\n"
    "Conflict reason : {reason}\n\n"
    "Write the status report back to Earth."
)

def _block_report(command: str, threat_type: str, sensors: dict, reason: str) -> str:
    if not _init_watsonx():
        return "(watsonx not configured — check .env credentials)"
    try:
        resp = _wx_model.chat(
            messages=[
                {"role": "system", "content": _BLOCK_SYSTEM},
                {"role": "user",   "content": _BLOCK_USER.format(
                    command=command, threat_type=threat_type,
                    sensors=str(sensors), reason=reason)},
            ],
            params={"max_tokens": 90, "temperature": 0.3, "repetition_penalty": 1.05},
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(model error: {e})"


def validate_command(command: str, sensor_state: dict,
                     threat_type: str | None = None,
                     comm_delay_s: float = 780) -> dict:
    """Validate an Earth command against current sensor state.

    Returns a dict with keys: verdict ('APPROVED'|'BLOCKED'), command,
    reason (str), earth_report (str).
    """
    cmd = command.strip().lower()

    def _approved():
        return {"verdict": "APPROVED", "command": command, "reason": "", "earth_report": ""}

    if not threat_type:
        return _approved()

    # cliff_edge — block advance commands when adj TTH <= RTT
    if threat_type == "cliff_edge" and cmd in _ADVANCE_CMDS:
        dist  = sensor_state.get("distance_m", float("inf"))
        speed = sensor_state.get("drift_speed_ms", 0.0)
        if speed > 0:
            tth_adj = (dist / speed) * THREAT_CONSERVATISM["cliff_edge"]
            rtt     = comm_delay_s * 2
            if tth_adj <= rtt:
                reason = (f"cliff edge {dist:.1f} m ahead; "
                          f"adj. time-to-edge {tth_adj:.0f} s ≤ round-trip {rtt:.0f} s")
                return {"verdict": "BLOCKED", "command": command, "reason": reason,
                        "earth_report": _block_report(command, threat_type, sensor_state, reason)}

    # dust_storm — block structural deployments when wind/opacity high
    if threat_type == "dust_storm" and cmd in _ANTENNA_CMDS:
        wind  = sensor_state.get("wind_speed_ms", 0.0)
        opdep = sensor_state.get("optical_depth", 0.0)
        if wind >= 15.0:
            reason = f"wind {wind:.1f} m/s exceeds structural safety limit (15 m/s)"
            return {"verdict": "BLOCKED", "command": command, "reason": reason,
                    "earth_report": _block_report(command, threat_type, sensor_state, reason)}
        if opdep >= 0.6:
            reason = f"dust optical depth {opdep:.3f} ≥ 0.60 — particulate ingestion risk"
            return {"verdict": "BLOCKED", "command": command, "reason": reason,
                    "earth_report": _block_report(command, threat_type, sensor_state, reason)}

    # battery_critical — block high-power commands when charge critically low
    if threat_type == "battery_critical" and cmd in _HIGH_PWR_CMDS:
        charge = sensor_state.get("charge_pct", 100.0)
        if charge <= 10.0:
            reason = f"battery at {charge:.1f}% — high-power command risks full shutdown"
            return {"verdict": "BLOCKED", "command": command, "reason": reason,
                    "earth_report": _block_report(command, threat_type, sensor_state, reason)}

    # rockfall — block any movement when debris ETA ≤ 30 s
    if threat_type == "rockfall" and cmd in _MOVEMENT_CMDS:
        dist  = sensor_state.get("debris_dist_m", float("inf"))
        speed = sensor_state.get("debris_speed_ms", 0.0)
        if speed > 0 and (dist / speed) <= 30.0:
            eta    = dist / speed
            reason = f"debris {dist:.1f} m at {speed:.1f} m/s — impact ETA {eta:.1f} s"
            return {"verdict": "BLOCKED", "command": command, "reason": reason,
                    "earth_report": _block_report(command, threat_type, sensor_state, reason)}

    # comms_blackout — block transmission commands when satellite below cutoff
    if threat_type == "comms_blackout" and cmd in _COMMS_CMDS:
        elev = sensor_state.get("relay_elevation_deg", 90.0)
        if elev <= 8.0:
            reason = f"relay at {elev:.1f}° (cutoff 8°) — transmission would fail"
            return {"verdict": "BLOCKED", "command": command, "reason": reason,
                    "earth_report": _block_report(command, threat_type, sensor_state, reason)}

    return _approved()


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _reset_sim():
    st.session_state.tick_ptr   = -1   # -1 = idle, 0..N = index of NEXT tick to process
    st.session_state.ticks_data = []
    st.session_state.history    = []
    st.session_state.log_feed   = []
    st.session_state.cmd_log    = []   # Earth Command history
    st.session_state.running    = False

if "tick_ptr" not in st.session_state:
    _reset_sim()
if "cmd_log" not in st.session_state:
    st.session_state.cmd_log = []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛰️ Sentinel Protocol")
    st.markdown("*Autonomous Decision-Time-Budget Engine*")
    st.divider()

    scenario = st.selectbox(
        "Threat Scenario",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda k: SCENARIO_LABELS[k],
        key="scenario_select",
    )

    comm_delay = st.slider(
        "One-way comm delay to Earth (s)",
        min_value=240, max_value=1440, value=780, step=60,
        help="Mars range: 240 s (near) → 1440 s (far). Default ≈ 13 min.",
    )

    n_ticks = st.slider("Simulation ticks", min_value=10, max_value=30, value=20)

    tick_delay = st.slider(
        "Delay between ticks (s)",
        min_value=0.3, max_value=3.0, value=1.0, step=0.1,
    )

    ai_on = st.toggle(
        "AI reasoning (watsonx.ai)", value=True,
        help="Calls ibm/granite-4-h-small for a mission-log sentence each tick.",
    )

    st.divider()

    col_run, col_rst = st.columns(2)
    with col_run:
        start_btn = st.button("▶ Run", type="primary", use_container_width=True)
    with col_rst:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    st.divider()
    st.caption(f"Round-trip: **{comm_delay * 2:,} s**")
    st.caption("GREEN > 2× RTT  |  YELLOW 1–2×  |  RED ≤ 1×")

# ─────────────────────────────────────────────────────────────────────────────
# Button handlers — mutate state, then fall through to render
# ─────────────────────────────────────────────────────────────────────────────

if reset_btn:
    _reset_sim()
    st.rerun()

if start_btn:
    _reset_sim()
    st.session_state.ticks_data = run_scenario(scenario, ticks=n_ticks, comm_delay_s=comm_delay)
    st.session_state.running    = True
    st.session_state.tick_ptr   = 0
    # Don't rerun here — fall through so the header renders before any tick work

# ─────────────────────────────────────────────────────────────────────────────
# Header (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0;'>🛰️ Sentinel Protocol</h1>"
    "<p style='color:#57606a;margin-top:4px;'>"
    "Autonomous Decision-Time-Budget Engine — Live Mission Dashboard</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Idle screen
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.tick_ptr == -1:
    st.info(
        "Select a threat scenario in the sidebar and press **▶ Run** to start.",
        icon="ℹ️",
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Advance one tick if running
# (happens silently — no UI yet — but ptr is updated before we draw)
# ─────────────────────────────────────────────────────────────────────────────

all_ticks: list[TickState] = st.session_state.ticks_data
ptr: int = st.session_state.tick_ptr

# Process the tick that ptr currently points at (if still running)
if st.session_state.running and ptr < len(all_ticks):
    state    = all_ticks[ptr]
    tier_val = state.tier.value

    conservatism = THREAT_CONSERVATISM[scenario]
    round_trip_s = comm_delay * 2
    adj_tth      = state.time_to_harm_s * conservatism
    ratio        = adj_tth / round_trip_s if round_trip_s > 0 else float("inf")

    if ai_on:
        ai_text = generate_reasoning(
            threat_type    = scenario,
            sensors_repr   = str(state.sensors),
            time_to_harm_s = state.time_to_harm_s,
            round_trip_s   = round_trip_s,
            ratio          = ratio,
            tier_val       = tier_val,
            action         = TIER_ACTION[tier_val],
        )
    else:
        ai_text = "(AI reasoning disabled)"

    ha_val = state.holding_action or ""
    st.session_state.history.append({
        "Tick":           ptr,
        "Tier":           f"{TIER_ICON[tier_val]} {tier_val}",
        "Holding Action": ha_val,
        "TTH (s)":        state.time_to_harm_s,
        "Adj Ratio":      round(ratio, 3),
        **state.sensors,
    })
    st.session_state.log_feed.append({
        "tick": ptr, "tier": tier_val, "text": ai_text,
    })

    # Advance pointer
    st.session_state.tick_ptr += 1
    if st.session_state.tick_ptr >= len(all_ticks):
        st.session_state.running = False

# ─────────────────────────────────────────────────────────────────────────────
# Derive display state from history (what we've processed so far)
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.history:
    # tick_ptr was just set to 0 by start_btn — nothing processed yet,
    # show a brief "starting" message then rerun to process tick 0
    st.info("Starting simulation…")
    time.sleep(0.3)
    st.rerun()

n_done      = len(st.session_state.history)
last        = st.session_state.history[-1]
last_state  = all_ticks[last["Tick"]]
last_tv     = last_state.tier.value

conservatism = THREAT_CONSERVATISM[scenario]
round_trip_s = comm_delay * 2
adj_tth      = last_state.time_to_harm_s * conservatism
ratio        = adj_tth / round_trip_s if round_trip_s > 0 else float("inf")

# ─────────────────────────────────────────────────────────────────────────────
# Progress + tier badge row
# ─────────────────────────────────────────────────────────────────────────────

prog_col, tier_col, tth_col, ratio_col = st.columns([2, 2, 1, 1])

with prog_col:
    label = "▶ running…" if st.session_state.running else "✅ complete"
    st.markdown(f"**Tick {n_done} / {len(all_ticks)}** — `{scenario}` — {label}")
    st.progress(n_done / len(all_ticks))

with tier_col:
    st.markdown(_tier_badge_html(last_tv, last_state.holding_action), unsafe_allow_html=True)

with tth_col:
    prev_tth = all_ticks[last["Tick"] - 1].time_to_harm_s if last["Tick"] > 0 else None
    delta    = f"{last_state.time_to_harm_s - prev_tth:+.0f} s" if prev_tth is not None else None
    st.metric("Time-to-Harm", f"{last_state.time_to_harm_s:,.0f} s", delta=delta)

with ratio_col:
    st.metric("Adj Ratio", f"{ratio:.3f}",
              help="> 2 = GREEN  |  1–2 = YELLOW  |  ≤ 1 = RED")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sensors + ratio bar  |  AI log feed
# ─────────────────────────────────────────────────────────────────────────────

sensor_col, log_col = st.columns([1, 2])

with sensor_col:
    st.markdown("#### 📡 Sensor Readings")
    for key, val in last_state.sensors.items():
        unit  = SENSOR_UNITS.get(key, "")
        label = key.replace("_", " ").title()
        st.metric(label, f"{val} {unit}".strip())

    st.markdown("#### Threat Budget Ratio")
    st.markdown(_ratio_bar_html(ratio), unsafe_allow_html=True)
    st.caption(f"RTT = {round_trip_s:,} s  ·  conservatism = {conservatism:.0%}")

with log_col:
    st.markdown("#### 🤖 AI Mission Log")
    if not st.session_state.log_feed:
        st.caption("No entries yet.")
    else:
        for entry in reversed(st.session_state.log_feed):
            t     = entry["tier"]
            tick  = entry["tick"]
            text  = entry["text"]
            color = TIER_COLOR[t]
            bg    = TIER_BG[t]
            icon  = TIER_ICON[t]
            st.markdown(
                f'<div style="border-left:4px solid {color};background:{bg};'
                f'border-radius:6px;padding:10px 14px;margin-bottom:8px;">'
                f'<span style="font-size:0.75rem;color:#57606a;">'
                f'Tick {tick} · {icon} {t}</span><br>'
                f'<span style="color:#1f2328;">{text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tick history table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("#### 📋 Tick History")

import pandas as pd

df = pd.DataFrame(st.session_state.history)

def _color_tier_cell(val: str):
    if "GREEN"  in val: return "background-color:#dcfce7;color:#16a34a;font-weight:600"
    if "YELLOW" in val: return "background-color:#fef9c3;color:#ca8a04;font-weight:600"
    if "RED"    in val: return "background-color:#fee2e2;color:#dc2626;font-weight:600"
    return ""

styled = (
    df.style
    .map(_color_tier_cell, subset=["Tier"])
    .format({"TTH (s)": "{:,.0f}", "Adj Ratio": "{:.3f}"})
)
st.dataframe(styled, use_container_width=True, height=min(60 + len(df) * 35, 400))

# ─────────────────────────────────────────────────────────────────────────────
# Earth Command Terminal
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("#### 📡 Earth Command Terminal")
st.caption(
    "Simulate a command arriving from Earth. The rover's AI validates it against "
    "the current sensor state before execution."
)

# Command options grouped by scenario relevance
_ALL_COMMANDS = [
    "move_forward",
    "continue_heading",
    "hold_position",
    "move_backward",
    "turn_left",
    "turn_right",
    "increase_speed",
    "transmit_data",
    "deploy_antenna",
    "run_diagnostics",
    "enable_heaters",
    "stop",
]

cmd_col, btn_col = st.columns([3, 1])

with cmd_col:
    selected_cmd = st.selectbox(
        "Command from Earth",
        options=_ALL_COMMANDS,
        key="earth_cmd_select",
        label_visibility="collapsed",
    )

with btn_col:
    send_btn = st.button("📤 Send", type="primary", use_container_width=True)

if send_btn:
    # Derive current threat and sensor state from the last processed tick
    current_sensors     = last_state.sensors
    current_threat      = scenario  # the active scenario is the active threat type
    current_tier        = last_tv   # string: "GREEN" / "YELLOW" / "RED"

    result = validate_command(
        command      = selected_cmd,
        sensor_state = current_sensors,
        threat_type  = current_threat,
        comm_delay_s = comm_delay,
    )

    # Store in command log (newest first prepend)
    st.session_state.cmd_log.insert(0, {
        "tick":    n_done,
        "tier":    current_tier,
        "cmd":     selected_cmd,
        "verdict": result["verdict"],
        "reason":  result["reason"],
        "report":  result["earth_report"],
    })

# Render the most recent command result prominently
if st.session_state.cmd_log:
    latest = st.session_state.cmd_log[0]
    verdict = latest["verdict"]

    if verdict == "BLOCKED":
        st.markdown(
            f'<div style="border:2px solid #dc2626;background:#fee2e2;border-radius:10px;'
            f'padding:16px 20px;margin-top:8px;">'
            f'<div style="font-size:1.1rem;font-weight:700;color:#dc2626;margin-bottom:6px;">'
            f'🚫 COMMAND BLOCKED — Tick {latest["tick"]} · 🔴 {latest["tier"]}</div>'
            f'<div style="color:#1f2328;margin-bottom:8px;">'
            f'<strong>Command:</strong> <code>{latest["cmd"]}</code></div>'
            f'<div style="color:#7f1d1d;font-size:0.88rem;margin-bottom:10px;">'
            f'<strong>Conflict:</strong> {latest["reason"]}</div>'
            f'<div style="border-top:1px solid #fca5a5;padding-top:10px;color:#1f2328;">'
            f'<strong>AI report to Earth</strong> '
            f'<span style="font-size:0.75rem;color:#57606a;">'
            f'(ibm/granite-4-h-small)</span><br>'
            f'<em>"{latest["report"]}"</em></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="border:2px solid #16a34a;background:#dcfce7;border-radius:10px;'
            f'padding:16px 20px;margin-top:8px;">'
            f'<div style="font-size:1.1rem;font-weight:700;color:#16a34a;margin-bottom:6px;">'
            f'✅ COMMAND APPROVED — Tick {latest["tick"]} · {TIER_ICON.get(latest["tier"], "")} {latest["tier"]}</div>'
            f'<div style="color:#1f2328;">'
            f'<strong>Command:</strong> <code>{latest["cmd"]}</code> — passed through to rover.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Scrollable command history (last 10)
    if len(st.session_state.cmd_log) > 1:
        with st.expander(f"Command history ({len(st.session_state.cmd_log)} sent)", expanded=False):
            for entry in st.session_state.cmd_log:
                v_color = "#dc2626" if entry["verdict"] == "BLOCKED" else "#16a34a"
                v_icon  = "🚫" if entry["verdict"] == "BLOCKED" else "✅"
                st.markdown(
                    f'<div style="border-left:3px solid {v_color};padding:6px 12px;'
                    f'margin-bottom:6px;background:#f7f8fa;border-radius:4px;">'
                    f'<span style="font-size:0.8rem;color:#57606a;">Tick {entry["tick"]} · {entry["tier"]}</span>'
                    f' &nbsp; {v_icon} <strong style="color:{v_color};">{entry["verdict"]}</strong>'
                    f' &nbsp; <code>{entry["cmd"]}</code>'
                    + (f'<br><span style="font-size:0.78rem;color:#57606a;">{entry["reason"]}</span>'
                       if entry["reason"] else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Schedule next tick — ALWAYS at the very bottom, AFTER all UI is drawn
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.running:
    time.sleep(tick_delay)
    st.rerun()
