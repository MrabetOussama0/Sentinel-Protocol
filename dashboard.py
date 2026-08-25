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

def run_scenario(threat_type: str, ticks: int, comm_delay_s: float) -> list[TickState]:
    """Pre-compute all ticks into a list (Streamlit-friendly — no generator in session state)."""
    results = []
    for i, (sensors, tth) in enumerate(_MODELS[threat_type](ticks)):
        tier = classify_threat(threat_type, tth, comm_delay_s)
        results.append(TickState(tick=i, sensors=sensors,
                                 time_to_harm_s=round(tth, 1), tier=tier))
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

def _tier_badge_html(tier_val: str) -> str:
    color = TIER_COLOR[tier_val]
    bg    = TIER_BG[tier_val]
    icon  = TIER_ICON[tier_val]
    return (
        f'<div style="background:{bg};border:2px solid {color};border-radius:12px;'
        f'padding:18px 28px;text-align:center;">'
        f'<span style="font-size:2.4rem;font-weight:800;color:{color};">'
        f'{icon} {tier_val}</span><br>'
        f'<span style="color:{color};font-size:0.85rem;">{TIER_ACTION[tier_val]}</span>'
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
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _reset_sim():
    st.session_state.tick_ptr   = -1   # -1 = idle, 0..N = index of NEXT tick to process
    st.session_state.ticks_data = []
    st.session_state.history    = []
    st.session_state.log_feed   = []
    st.session_state.running    = False

if "tick_ptr" not in st.session_state:
    _reset_sim()

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

    st.session_state.history.append({
        "Tick":      ptr,
        "Tier":      f"{TIER_ICON[tier_val]} {tier_val}",
        "TTH (s)":   state.time_to_harm_s,
        "Adj Ratio": round(ratio, 3),
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
    st.markdown(_tier_badge_html(last_tv), unsafe_allow_html=True)

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
# Schedule next tick — ALWAYS at the very bottom, AFTER all UI is drawn
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.running:
    time.sleep(tick_delay)
    st.rerun()
