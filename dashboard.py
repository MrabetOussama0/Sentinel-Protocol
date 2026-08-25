"""
Sentinel Protocol — Mission Control Dashboard
=============================================
Run with:  streamlit run dashboard.py

Six pre-scripted scenario stories covering the full decision lifecycle.
All logic lives in sentinel/; dashboard is a pure UI layer.
"""

import time
import math
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Sentinel Protocol · Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Package imports
# ─────────────────────────────────────────────────────────────────────────────
from sentinel.decision_engine import DecisionTier, THREAT_CONSERVATISM, classify_threat
from sentinel.simulator       import run_scenario, TickState
from sentinel.safety_gate     import validate_command, is_action_safe, blackout_survival_loop
from sentinel.reasoning       import generate_reasoning, make_block_report

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TC = {"GREEN": "#16a34a", "YELLOW": "#ca8a04", "RED": "#dc2626"}
BG = {"GREEN": "#dcfce7", "YELLOW": "#fef9c3", "RED": "#fee2e2"}
IC = {"GREEN": "🟢",      "YELLOW": "🟡",      "RED": "🔴"}
BD = {"GREEN": "#bbf7d0", "YELLOW": "#fde68a", "RED": "#fca5a5"}

SENSOR_UNITS = {
    "distance_m": "m", "drift_speed_ms": "m/s",
    "wind_speed_ms": "m/s", "dust_density_gcm3": "g/cm³", "optical_depth": "",
    "charge_pct": "%", "draw_pct_per_tick": "%/tick",
    "seismic_g": "g", "debris_dist_m": "m", "debris_speed_ms": "m/s",
    "relay_elevation_deg": "°", "effective_descent_rate": "°/tick",
}

SENSOR_ICONS = {
    "distance_m": "📏", "drift_speed_ms": "💨", "wind_speed_ms": "💨",
    "dust_density_gcm3": "🌫️", "optical_depth": "🌫️", "charge_pct": "🔋",
    "battery_pct": "🔋", "draw_pct_per_tick": "⚡", "seismic_g": "📳",
    "debris_dist_m": "🪨", "debris_speed_ms": "💥",
    "relay_elevation_deg": "📡", "effective_descent_rate": "📉",
}

SENSOR_UNITS = {
    **SENSOR_UNITS,
    "battery_pct": "%",
}

# State badge colours and icons
RS_COLOR = {
    "MOVING":        "#3b82d4",
    "HOLDING":       "#ca8a04",
    "STOPPED":       "#dc2626",
    "REPOSITIONING": "#7c5cd8",
    "CHARGING":      "#16a34a",
    "SURVIVAL":      "#dc2626",
}
RS_ICON = {
    "MOVING":        "🚗",
    "HOLDING":       "🛑",
    "STOPPED":       "⛔",
    "REPOSITIONING": "🔄",
    "CHARGING":      "⚡",
    "SURVIVAL":      "🛡️",
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-scripted scenario stories
# Each story has: title, icon, description, threat_type, comm_delay_s, ticks,
#   earth_response (what Earth replies at a given tick), earth_command (optional),
#   story_beats (list of (tick, narrative) annotations)
# ─────────────────────────────────────────────────────────────────────────────
STORIES = [
    {
        "id": "cliff_green_earth_ok",
        "icon": "🟢🪨",
        "title": "Cliff Edge — Earth Responds in Time",
        "subtitle": "GREEN tier · Earth command approved · Safe outcome",
        "description": (
            "Rover detects a cliff 100 m ahead while drifting slowly. "
            "Plenty of time before harm — system classifies GREEN and waits. "
            "Earth receives the alert and sends back 'hold_position'. "
            "The safety gate approves it. Rover halts safely."
        ),
        "threat": "cliff_edge",
        "comm_delay_s": 300,   # near-Mars — short delay, Earth has time
        "ticks": 18,
        "tick_delay": 0.7,
        "earth_cmd_at_tick": 2,          # Earth responds at tick 2
        "earth_cmd": "hold_position",    # safe command
        "story_beats": {
            0: "Cliff detected 100 m ahead. TTH well above RTT — GREEN tier.",
            2: "Earth receives alert. Sends 'hold_position' command.",
            3: "Safety gate approves command — cliff still at safe distance.",
            8: "Rover drifting closer. YELLOW tier reached — holding in place.",
            13: "RED tier — autonomous brake engaged. Rover halted safely.",
        },
    },
    {
        "id": "cliff_red_no_response",
        "icon": "🔴🪨",
        "title": "Cliff Edge — No Earth Response, Autonomous Action",
        "subtitle": "RED tier · Earth unreachable · Rover acts autonomously",
        "description": (
            "Rover drifts toward a cliff with a long comm delay (Mars far side). "
            "Time-to-harm drops below RTT before Earth can respond. "
            "System escalates to RED and autonomously executes emergency reverse. "
            "Earth is notified after the manoeuvre."
        ),
        "threat": "cliff_edge",
        "comm_delay_s": 780,
        "ticks": 18,
        "tick_delay": 0.7,
        "earth_cmd_at_tick": None,
        "earth_cmd": None,
        "story_beats": {
            0: "Cliff 100 m ahead. Comm delay = 26 min RTT. Monitoring.",
            2: "GREEN tier — TTH still comfortably above RTT.",
            8: "YELLOW tier — holding in place, Earth notified.",
            9: "RED tier — TTH < RTT. Cannot wait. Autonomous action initiated.",
            10: "Emergency reverse executed. Earth will be notified post-action.",
        },
    },
    {
        "id": "dust_storm_yellow_shelter",
        "icon": "🟡🌪️",
        "title": "Dust Storm — Hold and Reposition to Shelter",
        "subtitle": "GREEN → YELLOW · Earth sends reposition command · Approved",
        "description": (
            "Wind and dust rising steadily. System has time — GREEN initially. "
            "Earth is notified at YELLOW and sends 'move_backward' to reach shelter. "
            "Safety gate approves it (wind still below structural limit). "
            "Rover repositions before storm intensifies."
        ),
        "threat": "dust_storm",
        "comm_delay_s": 480,
        "ticks": 22,
        "tick_delay": 0.6,
        "earth_cmd_at_tick": 5,
        "earth_cmd": "move_backward",
        "story_beats": {
            0: "Trace dust detected. Wind 0.5 m/s. GREEN tier — no action.",
            3: "Storm building. YELLOW tier — reposition_to_safety holding action.",
            5: "Earth receives YELLOW alert. Sends 'move_backward' to shelter.",
            6: "Safety gate checks wind (< 15 m/s limit). Command APPROVED.",
            16: "RED tier — optical depth critical. Panels shielded autonomously.",
        },
    },
    {
        "id": "dust_storm_blocked_command",
        "icon": "🚫🪨",
        "title": "Cliff Edge — Earth Move Forward BLOCKED",
        "subtitle": "RED tier · Earth sends dangerous command · Blocked by safety gate",
        "description": (
            "Cliff 100 m ahead, rover drifting closer with a long comm delay. "
            "Earth — working from data captured before the hazard escalated — "
            "sends 'move_forward'. The safety gate calculates adj TTH is already "
            "below RTT: executing would risk driving over the edge. Blocked. "
            "AI-generated explanation sent back to Earth."
        ),
        "threat": "cliff_edge",
        "comm_delay_s": 780,
        "ticks": 18,
        "tick_delay": 0.7,
        "earth_cmd_at_tick": 10,
        "earth_cmd": "move_forward",
        "story_beats": {
            0: "Cliff 100 m ahead. GREEN tier — monitoring.",
            8: "YELLOW — holding in place, Earth notified.",
            9: "RED tier — adj TTH < RTT. Autonomous brake engaged.",
            10: "Earth (stale data) sends 'move_forward' — BLOCKED: adj TTH < RTT.",
            11: "AI report to Earth: cliff hazard prevents execution of command.",
        },
    },
    {
        "id": "battery_critical_rescue",
        "icon": "🔋⚡",
        "title": "Battery Critical — Autonomous Solar Rescue",
        "subtitle": "GREEN → YELLOW → RED · Autonomous reposition to sunlight",
        "description": (
            "Battery draining with no sunlight. System monitors TTH vs RTT. "
            "At YELLOW, rover begins repositioning toward a known sunlit area. "
            "At RED, Earth is unreachable — rover autonomously navigates to "
            "charging position and enters low-power mode."
        ),
        "threat": "battery_critical",
        "comm_delay_s": 780,
        "ticks": 20,
        "tick_delay": 0.7,
        "earth_cmd_at_tick": None,
        "earth_cmd": None,
        "story_beats": {
            0: "Battery at 30%, draw rate 0.3%/tick. GREEN tier.",
            4: "GREEN — Earth notified of battery trend.",
            5: "YELLOW — reposition_to_safety holding action started.",
            11: "Draw accelerating. YELLOW still holding.",
            12: "RED — battery below critical threshold. Autonomous low-power mode.",
        },
    },
    {
        "id": "comms_blackout_survival",
        "icon": "📡🌑",
        "title": "Comms Blackout — Survival Loop",
        "subtitle": "GREEN → RED · Full communications lost · Autonomous survival",
        "description": (
            "Relay satellite descending toward horizon — contact window closing. "
            "System escalates through GREEN → YELLOW → RED as relay elevation drops. "
            "At RED, total blackout: rover enters the survival loop — hold, check "
            "for stacked hazards, monitor battery, queue logs for when contact resumes."
        ),
        "threat": "comms_blackout",
        "comm_delay_s": 780,
        "ticks": 20,
        "tick_delay": 0.7,
        "earth_cmd_at_tick": 1,
        "earth_cmd": "run_diagnostics",  # approved while still GREEN
        "story_beats": {
            0: "Relay at 80°. GREEN tier — full contact.",
            1: "Earth sends 'run_diagnostics' — approved while GREEN.",
            4: "YELLOW tier — relay descending, holding and monitoring.",
            12: "YELLOW — relay at 62°, descent accelerating.",
            13: "RED — relay below threshold. Blackout imminent. Survival mode.",
        },
    },
    # ── NEW: stays YELLOW, never reaches RED ─────────────────────────────
    {
        "id": "dust_storm_yellow_hold",
        "icon": "🟡☁️",
        "title": "Dust Storm — Holds at YELLOW, Never RED",
        "subtitle": "GREEN → YELLOW only · Storm subsides · Earth responds in time",
        "description": (
            "A gentle, slow-building dust storm raises concern but the communication "
            "delay to Earth is short enough that TTH never drops below RTT. "
            "The rover never needs to act autonomously — it holds at YELLOW, "
            "Earth responds, and the storm eventually passes. "
            "This scenario shows the tier progression is NOT always GREEN→YELLOW→RED."
        ),
        "threat": "dust_storm_slow",
        "comm_delay_s": 2500,  # Earth far away — TTH never drops below RTT, stays YELLOW
        "ticks": 25,
        "tick_delay": 0.5,
        "earth_cmd_at_tick": 8,
        "earth_cmd": "hold_position",
        "story_beats": {
            0:  "Trace dust, light breeze. GREEN — science ops continue.",
            4:  "Wind building gradually. Still GREEN.",
            5:  "YELLOW threshold reached — repositioning to shelter, Earth notified.",
            8:  "Earth responds (within budget window): 'hold_position' sent.",
            9:  "Safety gate approves — wind below structural limit. Rover holds.",
            15: "Storm still YELLOW — rover maintaining safe hold position.",
            24: "Scenario ends at YELLOW. No autonomous RED action needed.",
        },
    },
    # ── NEW: full blackout — no Earth contact, rover self-manages ─────────
    {
        "id": "full_blackout_survival",
        "icon": "🌑🤖",
        "title": "Full Blackout — Rover Self-Manages Until Contact Returns",
        "subtitle": "No Earth contact · Dust + battery + cliff · Autonomous survival",
        "description": (
            "The relay satellite went below the horizon BEFORE this scenario starts — "
            "there is no Earth contact from tick 0. The rover faces rising dust, "
            "a draining battery, and a cliff ahead, all simultaneously. "
            "With no commands possible, it stops, shields against dust, conserves "
            "power, holds its position, and waits for a rescue relay to rise "
            "over the horizon at tick ~14. This is pure autonomous survival."
        ),
        "threat": "full_blackout",
        "comm_delay_s": 780,
        "ticks": 22,
        "tick_delay": 0.7,
        "earth_cmd_at_tick": None,
        "earth_cmd": None,
        "story_beats": {
            0:  "Relay below horizon — BLACKOUT. No Earth contact. Survival mode begins.",
            2:  "Battery draining. Dust rising. Cliff 180 m ahead. Holding position.",
            5:  "Non-essential systems powered down. Wind 5 m/s and climbing.",
            8:  "Battery at ~48%. Wind 7.4 m/s. Instruments stowed. Still holding.",
            10: "Wind 9 m/s. Battery 40%. Dust shields deployed. Awaiting relay.",
            14: "Rescue relay rising over horizon. Contact window approaching.",
            17: "Relay at 6°. Battery 28%. Attempting to queue transmission.",
            21: "Relay re-established. Survival logs transmitted. Rover safe.",
        },
    },
]

STORY_BY_ID = {s["id"]: s for s in STORIES}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_reasoning(threat_type, sensors_repr, tth, rtt, ratio, tier_val, action):
    return generate_reasoning({
        "threat_type": threat_type, "sensors": sensors_repr,
        "time_to_harm_s": tth, "round_trip_s": rtt,
        "ratio": ratio, "tier": tier_val, "action": action,
    })

@st.cache_data(show_spinner=False)
def _run_ticks(threat: str, ticks: int, comm_delay: float):
    return list(run_scenario(threat, ticks=ticks, comm_delay_s=comm_delay))

def _tier_css(t):
    return f"background:{BG[t]};border:2px solid {TC[t]};border-radius:10px;padding:14px 20px;"

def _ratio_bar_html(ratio: float) -> str:
    pct = min(ratio / 3.0, 1.0) * 100
    tv  = "GREEN" if ratio > 2 else ("YELLOW" if ratio > 1 else "RED")
    return (
        f'<div style="background:#e5e7eb;border-radius:8px;height:18px;width:100%;position:relative;">'
        f'<div style="background:{TC[tv]};width:{pct:.1f}%;height:18px;border-radius:8px;'
        f'transition:width 0.4s ease;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#6b7280;margin-top:3px;">'
        f'<span>0</span><span style="margin-left:33%">1× RTT</span>'
        f'<span style="margin-left:auto">2× RTT</span><span>3×</span></div>'
    )

def _tier_big_html(tier_val: str, action_line: str, holding: str | None = None) -> str:
    c, b, bd = TC[tier_val], BG[tier_val], BD[tier_val]
    hold_line = ""
    if holding:
        hico = "🧲" if holding == "hold_in_place" else "🔄"
        hold_line = (
            f'<div style="margin-top:8px;display:inline-block;background:{bd};'
            f'border-radius:6px;padding:4px 12px;font-size:0.82rem;color:{c};">'
            f'{hico} {holding.replace("_", " ")}</div>'
        )
    return (
        f'<div style="{_tier_css(tier_val)};text-align:center;">'
        f'<div style="font-size:3rem;font-weight:900;color:{c};line-height:1;">{IC[tier_val]}</div>'
        f'<div style="font-size:1.5rem;font-weight:800;color:{c};margin-top:4px;">{tier_val}</div>'
        f'<div style="font-size:0.85rem;color:{c};margin-top:6px;opacity:0.9;">{action_line}</div>'
        f'{hold_line}</div>'
    )

def _robot_status_html(robot_state: str, robot_activity: str,
                        prev_state: str | None = None) -> str:
    c   = RS_COLOR.get(robot_state, "#57606a")
    ico = RS_ICON.get(robot_state, "🤖")
    changed = prev_state is not None and prev_state != robot_state
    border  = f"2px solid {c}" if not changed else f"3px solid {c}"
    badge_bg = c + "22"
    change_note = (
        f'<div style="font-size:0.72rem;color:{c};margin-top:4px;font-weight:600;">'
        f'◀ Changed from {prev_state}</div>'
        if changed else ""
    )
    return (
        f'<div style="background:#fff;{border};border-radius:12px;padding:14px 18px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'<div style="background:{badge_bg};border-radius:8px;padding:6px 14px;'
        f'font-size:0.85rem;font-weight:800;color:{c};letter-spacing:0.05em;">'
        f'{ico}&nbsp; {robot_state}</div>'
        f'<div style="font-size:0.72rem;color:#57606a;margin-left:auto;">Rover Status</div>'
        f'</div>'
        f'<div style="font-size:0.9rem;color:#1f2328;line-height:1.5;">{robot_activity}</div>'
        f'{change_note}</div>'
    )

def _sensor_card_html(key: str, val, unit: str) -> str:
    icon = SENSOR_ICONS.get(key, "📊")
    label = key.replace("_", " ").title()
    display = f"{val} {unit}".strip() if not isinstance(val, float) else f"{val:.3g} {unit}".strip()
    return (
        f'<div style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:12px 14px;margin-bottom:8px;">'
        f'<div style="font-size:0.72rem;color:#57606a;">{icon} {label}</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:#1f2328;margin-top:2px;">{display}</div>'
        f'</div>'
    )

def _event_pill(color: str, text: str) -> str:
    return (
        f'<span style="display:inline-block;background:{color}22;border:1px solid {color};'
        f'color:{color};border-radius:20px;padding:2px 10px;font-size:0.72rem;'
        f'font-weight:600;margin:2px 4px 2px 0;">{text}</span>'
    )

def _log_entry_html(tick: int, tier: str, source: str, text: str,
                    cmd_verdict: str = "", cmd: str = "", beat: str = "",
                    robot_state: str = "") -> str:
    c, b = TC[tier], BG[tier]
    src_icons = {"EMERGENCY": "🚨", "AUTONOMOUS": "🤖", "EARTH": "📡", "IDLE": "💤", "": ""}
    pills = f'<div style="margin-top:4px;">'
    if robot_state:
        rs_c = RS_COLOR.get(robot_state, "#57606a")
        rs_i = RS_ICON.get(robot_state, "🤖")
        pills += _event_pill(rs_c, f"{rs_i} {robot_state}")
    if cmd_verdict == "APPROVED":
        pills += _event_pill("#16a34a", f"✅ Earth cmd: {cmd}")
    elif cmd_verdict == "BLOCKED":
        pills += _event_pill("#dc2626", f"🚫 Blocked: {cmd}")
    if beat:
        pills += _event_pill("#3b82d4", f"📋 {beat}")
    pills += "</div>"
    return (
        f'<div style="border-left:4px solid {c};background:{b};'
        f'border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:0.75rem;color:#57606a;">'
        f'Tick {tick:02d} &nbsp;·&nbsp; {IC[tier]} {tier} &nbsp;·&nbsp; '
        f'{src_icons.get(source,"")} {source}</span></div>'
        f'<div style="color:#1f2328;margin-top:5px;font-size:0.9rem;">{text}</div>'
        f'{pills}</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _reset(story_id: str | None = None):
    st.session_state.story_id      = story_id
    st.session_state.tick_ptr      = -1
    st.session_state.ticks_data    = []
    st.session_state.history_rows  = []
    st.session_state.log_feed      = []
    st.session_state.cmd_result    = None
    st.session_state.cmd_fired     = False
    st.session_state.running       = False
    st.session_state.blackout_steps= []

if "story_id" not in st.session_state:
    _reset()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
  <div style="font-size:2.8rem;line-height:1;">🛰️</div>
  <div>
    <div style="font-size:1.6rem;font-weight:900;color:#1f2328;line-height:1.1;">
      Sentinel Protocol</div>
    <div style="font-size:0.88rem;color:#57606a;">
      Autonomous Decision-Time-Budget Engine · Planetary Rover Mission Control</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SELECTOR  (always visible at top)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎬 Mission Scenarios")
st.markdown(
    "<p style='color:#57606a;font-size:0.9rem;margin-top:-8px;'>"
    "Each scenario is a complete, realistic mission event — select one to run it live.</p>",
    unsafe_allow_html=True,
)

cols = st.columns(3)
for idx, story in enumerate(STORIES):
    col = cols[idx % 3]
    with col:
        active = st.session_state.story_id == story["id"]
        border = f"3px solid {TC['GREEN']}" if active else "2px solid #e5e7eb"
        bg     = "#f0fdf4" if active else "#fafafa"
        st.markdown(
            f'<div style="border:{border};border-radius:12px;background:{bg};'
            f'padding:14px 16px;margin-bottom:4px;min-height:120px;">'
            f'<div style="font-size:1.5rem;">{story["icon"]}</div>'
            f'<div style="font-weight:700;font-size:0.95rem;color:#1f2328;margin-top:4px;">'
            f'{story["title"]}</div>'
            f'<div style="font-size:0.75rem;color:#57606a;margin-top:3px;">'
            f'{story["subtitle"]}</div></div>',
            unsafe_allow_html=True,
        )
        label = "▶  Running…" if active and st.session_state.running else (
                "✅ View Results" if active and not st.session_state.running and st.session_state.tick_ptr > 0
                else "▶  Launch")
        btn_type = "primary" if active else "secondary"
        if st.button(label, key=f"btn_{story['id']}", use_container_width=True, type=btn_type):
            if not active or not st.session_state.running:
                _reset(story["id"])
                story_cfg = STORY_BY_ID[story["id"]]
                st.session_state.ticks_data = _run_ticks(
                    story_cfg["threat"], story_cfg["ticks"], story_cfg["comm_delay_s"]
                )
                st.session_state.running  = True
                st.session_state.tick_ptr = 0
                st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Nothing selected yet
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.story_id is None or not st.session_state.ticks_data:
    st.markdown("""
    <div style="text-align:center;padding:48px 0;color:#57606a;">
      <div style="font-size:3rem;">🛰️</div>
      <div style="font-size:1.1rem;font-weight:600;margin-top:12px;color:#1f2328;">
        Select a scenario above to begin</div>
      <div style="font-size:0.88rem;margin-top:6px;">
        Each scenario plays out tick by tick with live sensor data, decision tiers,<br>
        Earth command interactions, and AI-generated mission log entries.</div>
    </div>
    """, unsafe_allow_html=True)

    # How it works explainer
    st.markdown("### How Sentinel Protocol Works")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown("""
        **📊 Decision-Time-Budget Engine**
        
        For every detected hazard, Sentinel estimates *time-to-harm* (TTH) 
        and compares it to the *round-trip communication delay* (RTT) to Earth.
        
        - **TTH > 2× RTT** → 🟢 GREEN — wait for Earth  
        - **TTH 1–2× RTT** → 🟡 YELLOW — hold and notify  
        - **TTH ≤ RTT** → 🔴 RED — act now, notify after
        """)
    with ec2:
        st.markdown("""
        **🛡️ Universal Safety Gate**
        
        Every action — whether sent by Earth or generated by the rover — 
        passes through `is_action_safe()` before execution.
        
        - Earth commands are **validated against live sensors**
        - Outdated or unsafe commands are **blocked with an AI explanation**
        - No action executes on an assumption — every step is checked
        """)
    with ec3:
        st.markdown("""
        **🤖 AI Reasoning Layer**
        
        IBM watsonx.ai (`granite-4-h-small`) generates a plain-language 
        mission log entry for every decision tick.
        
        - Explains *why* the tier was assigned
        - Written in the voice of a flight engineer
        - Mission controllers can audit every autonomous action
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Active story — advance one tick
# ─────────────────────────────────────────────────────────────────────────────
story_cfg  = STORY_BY_ID[st.session_state.story_id]
all_ticks: list[TickState] = st.session_state.ticks_data
ptr        = st.session_state.tick_ptr
comm_delay = story_cfg["comm_delay_s"]
threat     = story_cfg["threat"]
rtt        = comm_delay * 2
conservatism = THREAT_CONSERVATISM[threat]
tick_delay = story_cfg.get("tick_delay", 0.8)

TIER_ACTION_SHORT = {
    "GREEN":  "Waiting for Earth response",
    "YELLOW": "Holding — safe action active, Earth notified",
    "RED":    "Acting autonomously — notifying Earth after",
}

if st.session_state.running and ptr < len(all_ticks):
    ts: TickState = all_ticks[ptr]
    tv = ts.tier.value
    adj_tth = ts.time_to_harm_s * conservatism
    ratio   = adj_tth / rtt if rtt > 0 else float("inf")

    # Earth command interaction
    cmd_verdict, cmd_text, cmd_report = "", "", ""
    if story_cfg["earth_cmd_at_tick"] == ptr and story_cfg["earth_cmd"] and not st.session_state.cmd_fired:
        earth_cmd = story_cfg["earth_cmd"]
        result = validate_command(
            command          = earth_cmd,
            sensor_state     = ts.sensors,
            threat_type      = threat,
            comm_delay_s     = comm_delay,
            _block_report_fn = make_block_report,
        )
        cmd_verdict = result.verdict
        cmd_text    = earth_cmd
        cmd_report  = result.earth_report
        st.session_state.cmd_result  = result
        st.session_state.cmd_fired   = True

    # Blackout survival loop at RED for comms_blackout story
    if threat == "comms_blackout" and tv == "RED" and not st.session_state.blackout_steps:
        st.session_state.blackout_steps = list(blackout_survival_loop(
            sensor_state=ts.sensors, comm_delay_s=comm_delay
        ))

    # AI reasoning
    ai_text = _cached_reasoning(
        threat_type  = threat,
        sensors_repr = str(ts.sensors),
        tth          = ts.time_to_harm_s,
        rtt          = rtt,
        ratio        = ratio,
        tier_val     = tv,
        action       = TIER_ACTION_SHORT[tv],
    )

    beat = story_cfg["story_beats"].get(ptr, "")
    st.session_state.log_feed.append({
        "tick": ptr, "tier": tv, "tth": ts.time_to_harm_s,
        "ratio": ratio, "sensors": ts.sensors,
        "holding": ts.holding_action, "ai": ai_text, "beat": beat,
        "cmd_verdict": cmd_verdict, "cmd_text": cmd_text, "cmd_report": cmd_report,
        "robot_state": ts.robot_state, "robot_activity": ts.robot_activity,
    })
    st.session_state.history_rows.append({
        "Tick": ptr, "Tier": f"{IC[tv]} {tv}",
        "TTH (s)": round(ts.time_to_harm_s), "Ratio": round(ratio, 3),
        "Action": ts.holding_action or ("autonomous" if tv == "RED" else "—"),
        "Earth Cmd": cmd_text or "—", "Verdict": cmd_verdict or "—",
        **{k: v for k, v in ts.sensors.items()},
    })
    st.session_state.tick_ptr += 1
    if st.session_state.tick_ptr >= len(all_ticks):
        st.session_state.running = False

# Guard: nothing rendered yet
if not st.session_state.log_feed:
    st.info("Starting scenario…")
    time.sleep(0.2)
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# RENDER current state
# ─────────────────────────────────────────────────────────────────────────────
latest      = st.session_state.log_feed[-1]
last_ptr    = latest["tick"]
last_ts     = all_ticks[last_ptr]
tv          = latest["tier"]
ratio       = latest["ratio"]
tth         = latest["tth"]
n_done      = len(st.session_state.log_feed)
total_ticks = len(all_ticks)
complete    = not st.session_state.running

# ── Story header ─────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
    f'<div style="font-size:2rem;">{story_cfg["icon"]}</div>'
    f'<div><div style="font-size:1.25rem;font-weight:800;color:#1f2328;">'
    f'{story_cfg["title"]}</div>'
    f'<div style="font-size:0.85rem;color:#57606a;">{story_cfg["subtitle"]}</div></div>'
    f'<div style="margin-left:auto;font-size:0.8rem;color:#57606a;">'
    f'RTT = {rtt:,} s &nbsp;·&nbsp; conservatism = {conservatism:.0%}</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;'
    f'padding:10px 14px;font-size:0.88rem;color:#57606a;margin-bottom:12px;">'
    f'ℹ️ {story_cfg["description"]}</div>',
    unsafe_allow_html=True,
)

# ── Progress bar ─────────────────────────────────────────────────────────────
prog_label = "✅ Complete" if complete else f"▶ Tick {n_done} / {total_ticks}"
st.progress(n_done / total_ticks, text=prog_label)

# ─────────────────────────────────────────────────────────────────────────────
# ROVER STATUS CARD — what is the robot doing RIGHT NOW
# ─────────────────────────────────────────────────────────────────────────────
robot_state    = latest.get("robot_state", "MOVING")
robot_activity = latest.get("robot_activity", "")
prev_robot_state = (
    st.session_state.log_feed[-2].get("robot_state")
    if len(st.session_state.log_feed) > 1 else None
)
st.markdown("#### 🤖 Rover Status — What is the robot doing right now?")
st.markdown(
    _robot_status_html(robot_state, robot_activity, prev_robot_state),
    unsafe_allow_html=True,
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN GRID: Tier card | Sensors | Ratio bar
# ─────────────────────────────────────────────────────────────────────────────
tier_col, sensor_col, ratio_col = st.columns([1.6, 1.4, 2])

with tier_col:
    st.markdown("#### Decision Tier")
    st.markdown(
        _tier_big_html(tv, TIER_ACTION_SHORT[tv], latest["holding"]),
        unsafe_allow_html=True,
    )

    # Key metrics
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    prev_tth = all_ticks[last_ptr - 1].time_to_harm_s if last_ptr > 0 else None
    delta_tth = f"{tth - prev_tth:+,.0f} s" if prev_tth else None
    m1.metric("Time-to-Harm", f"{tth:,.0f} s", delta=delta_tth,
              delta_color="inverse")
    m2.metric("Adj. Ratio", f"{ratio:.3f}",
              help="> 2 = GREEN · 1–2 = YELLOW · ≤ 1 = RED")

    m3, m4 = st.columns(2)
    m3.metric("One-way delay", f"{comm_delay:,} s")
    m4.metric("RTT", f"{rtt:,} s")

with sensor_col:
    st.markdown("#### 📡 Live Sensors")
    for k, v in last_ts.sensors.items():
        unit = SENSOR_UNITS.get(k, "")
        st.markdown(_sensor_card_html(k, v, unit), unsafe_allow_html=True)

with ratio_col:
    st.markdown("#### 📊 Threat Budget Ratio")
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;'
        f'padding:16px 20px;">'
        f'<div style="font-size:0.8rem;color:#57606a;margin-bottom:8px;">'
        f'Adjusted TTH ÷ RTT &nbsp;(conservatism {conservatism:.0%} applied)</div>'
        + _ratio_bar_html(ratio)
        + f'<div style="font-size:2rem;font-weight:800;color:{TC[tv]};margin-top:12px;">'
        f'{ratio:.3f}</div>'
        f'<div style="font-size:0.8rem;color:{TC[tv]};">'
        f'{"SAFE TO WAIT" if tv=="GREEN" else ("HOLD & NOTIFY" if tv=="YELLOW" else "ACT NOW")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Earth command result card (shown when it fires)
    cmd_res = st.session_state.cmd_result
    if cmd_res:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📡 Earth Command")
        if cmd_res.verdict == "APPROVED":
            st.markdown(
                f'<div style="background:#dcfce7;border:2px solid #16a34a;'
                f'border-radius:10px;padding:14px 16px;">'
                f'<div style="font-weight:700;color:#15803d;font-size:1rem;">✅ APPROVED</div>'
                f'<div style="margin-top:6px;color:#1f2328;">'
                f'<b>Command:</b> <code>{cmd_res.command}</code></div>'
                f'<div style="font-size:0.8rem;color:#57606a;margin-top:4px;">'
                f'Passed safety gate — executing.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#fee2e2;border:2px solid #dc2626;'
                f'border-radius:10px;padding:14px 16px;">'
                f'<div style="font-weight:700;color:#dc2626;font-size:1rem;">🚫 BLOCKED</div>'
                f'<div style="margin-top:6px;color:#1f2328;">'
                f'<b>Command:</b> <code>{cmd_res.command}</code></div>'
                f'<div style="font-size:0.82rem;color:#7f1d1d;margin-top:4px;">'
                f'<b>Reason:</b> {cmd_res.reason}</div>'
                + (f'<div style="border-top:1px solid #fca5a5;margin-top:8px;padding-top:8px;'
                   f'font-size:0.8rem;color:#1f2328;"><b>AI report to Earth:</b><br>'
                   f'<em>{cmd_res.earth_report}</em></div>' if cmd_res.earth_report else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE (tick-by-tick event log)
# ─────────────────────────────────────────────────────────────────────────────
log_col, table_col = st.columns([1.4, 1.6])

with log_col:
    st.markdown("#### 🗓️ Mission Timeline")
    # Show latest first
    for entry in reversed(st.session_state.log_feed):
        src = ("EMERGENCY" if entry["tier"] == "RED" else
               ("AUTONOMOUS" if entry["holding"] else "IDLE"))
        st.markdown(
            _log_entry_html(
                tick         = entry["tick"],
                tier         = entry["tier"],
                source       = src,
                text         = entry["ai"],
                cmd_verdict  = entry["cmd_verdict"],
                cmd          = entry["cmd_text"],
                beat         = entry["beat"],
                robot_state  = entry.get("robot_state", ""),
            ),
            unsafe_allow_html=True,
        )

with table_col:
    st.markdown("#### 📋 Tick History")
    df = pd.DataFrame(st.session_state.history_rows)

    def _ct(val):
        v = str(val)
        if "GREEN"  in v: return "background-color:#dcfce7;color:#16a34a;font-weight:700"
        if "YELLOW" in v: return "background-color:#fef9c3;color:#ca8a04;font-weight:700"
        if "RED"    in v: return "background-color:#fee2e2;color:#dc2626;font-weight:700"
        if "BLOCKED" in v: return "background-color:#fee2e2;color:#dc2626"
        if "APPROVED" in v: return "background-color:#dcfce7;color:#16a34a"
        return ""

    base_cols = ["Tick", "Tier", "TTH (s)", "Ratio", "Action", "Earth Cmd", "Verdict"]
    extra = [c for c in df.columns if c not in base_cols]
    disp  = base_cols + extra
    styled = (
        df[disp].style
        .map(_ct, subset=["Tier", "Verdict"])
        .format({"TTH (s)": "{:,}", "Ratio": "{:.3f}"}, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=min(80 + len(df) * 35, 520))

    # Tier distribution summary
    if complete and len(df) > 0:
        st.markdown("##### Summary")
        tier_counts = df["Tier"].str.extract(r"(GREEN|YELLOW|RED)")[0].value_counts()
        sc1, sc2, sc3 = st.columns(3)
        for col, t in zip([sc1, sc2, sc3], ["GREEN", "YELLOW", "RED"]):
            n = tier_counts.get(t, 0)
            col.markdown(
                f'<div style="background:{BG[t]};border:1px solid {TC[t]};border-radius:8px;'
                f'padding:10px;text-align:center;">'
                f'<div style="font-size:1.5rem;font-weight:800;color:{TC[t]};">{n}</div>'
                f'<div style="font-size:0.75rem;color:{TC[t]};">{IC[t]} {t} ticks</div></div>',
                unsafe_allow_html=True,
            )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# BLACKOUT SURVIVAL LOOP (shown for comms_blackout story when reached RED)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.blackout_steps:
    st.markdown("#### 🌑 Blackout Survival Loop")
    st.markdown(
        "<p style='color:#57606a;font-size:0.88rem;margin-top:-8px;'>"
        "Total comms lost. Every action re-validated through the safety gate before execution.</p>",
        unsafe_allow_html=True,
    )
    bs_cols = st.columns(min(len(st.session_state.blackout_steps), 4))
    for i, step in enumerate(st.session_state.blackout_steps):
        col = bs_cols[i % len(bs_cols)]
        ok_color = TC["GREEN"] if step.executed else TC["RED"]
        ok_icon  = "✅" if step.executed else "🚫"
        col.markdown(
            f'<div style="background:#f7f8fa;border:2px solid {ok_color};'
            f'border-radius:10px;padding:12px;text-align:center;height:120px;">'
            f'<div style="font-size:1.5rem;">{ok_icon}</div>'
            f'<div style="font-weight:700;font-size:0.82rem;color:{ok_color};margin-top:4px;">'
            f'[{step.phase}]</div>'
            f'<div style="font-size:0.75rem;color:#1f2328;margin-top:4px;">'
            f'<code>{step.proposed}</code></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL COMMAND TERMINAL (always visible when a scenario is running/complete)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📡 Send a Custom Earth Command", expanded=False):
    st.markdown(
        "Test any command against the rover's **current** sensor state and active threat. "
        "The safety gate evaluates it in real time."
    )
    ALL_COMMANDS = [
        "move_forward", "continue_heading", "increase_speed",
        "move_backward", "reverse", "turn_left", "turn_right",
        "hold_position", "stop", "transmit_data", "deploy_antenna",
        "run_diagnostics", "enable_heaters", "activate_drill",
    ]
    mc1, mc2 = st.columns([3, 1])
    with mc1:
        custom_cmd = st.selectbox("Command", ALL_COMMANDS, key="custom_cmd",
                                  label_visibility="collapsed")
    with mc2:
        send_custom = st.button("📤 Send", type="primary",
                                use_container_width=True, key="send_custom")
    if send_custom:
        cr = validate_command(
            command=custom_cmd, sensor_state=last_ts.sensors,
            threat_type=threat, comm_delay_s=comm_delay,
            _block_report_fn=make_block_report,
        )
        if cr.verdict == "APPROVED":
            st.success(f"✅ **APPROVED** — `{custom_cmd}` passed the safety gate and would execute.")
        else:
            st.error(f"🚫 **BLOCKED** — `{custom_cmd}` rejected: {cr.reason}")
            if cr.earth_report:
                st.markdown(f"**AI report to Earth:** *{cr.earth_report}*")

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY GATE PROBE
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍 Safety Gate Probe — Test Any Action vs. Any Threat", expanded=False):
    st.markdown(
        "Call `is_action_safe()` directly with custom inputs. "
        "Useful for exploring edge cases."
    )
    pg1, pg2 = st.columns(2)
    with pg1:
        probe_action = st.selectbox(
            "Action", ALL_COMMANDS + ["emergency_full_stop", "cut_motors"], key="probe_action"
        )
        probe_threats = st.multiselect(
            "Active threats",
            ["cliff_edge", "dust_storm", "battery_critical",
             "rockfall", "comms_blackout", "unclassified_anomaly"],
            default=[threat], key="probe_threats",
        )
    with pg2:
        p_dist   = st.number_input("distance_m",    0.0, 5000.0, 50.0,  key="p_dist")
        p_charge = st.number_input("charge_pct",    0.0,  100.0, 30.0,  key="p_charge")
        p_wind   = st.number_input("wind_speed_ms", 0.0,   60.0,  5.0,  key="p_wind")
        p_debris = st.number_input("debris_dist_m", 0.0, 5000.0, 500.0, key="p_debris")
    if st.button("🔎 Check", key="probe_btn"):
        ps = {
            "distance_m": p_dist, "drift_speed_ms": 0.05,
            "charge_pct": p_charge, "draw_pct_per_tick": 0.5,
            "wind_speed_ms": p_wind, "optical_depth": 0.1,
            "debris_dist_m": p_debris, "debris_speed_ms": 1.0,
            "relay_elevation_deg": 30.0,
        }
        pr = is_action_safe(probe_action, ps, probe_threats, comm_delay)
        if pr.safe:
            st.success(f"✅ **SAFE** — `{probe_action}` is permitted with threats: {probe_threats}")
        else:
            st.error(
                f"🚫 **BLOCKED** by `{pr.blocked_by}` — `{probe_action}`\n\n"
                f"**Reason:** {pr.reason}"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TICK ADVANCE — always at bottom
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(tick_delay)
    st.rerun()
