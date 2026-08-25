"""
Sentinel Protocol — Live Mission Dashboard
==========================================
Run with:  streamlit run dashboard.py

Architecture
------------
Every Streamlit render pass:
  1. Advances one tick through the DecisionManager (the central arbitrator).
  2. Renders the full UI — tier badge, sensor panel, AI log, tick history,
     Earth command terminal, anomaly panel, blackout survival panel.
  3. Schedules the next tick via st.rerun() at the very bottom.

All decision logic lives in sentinel/; the dashboard is a pure UI layer.
"""

import time
import streamlit as st

st.set_page_config(
    page_title="Sentinel Protocol",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# sentinel package imports
# ─────────────────────────────────────────────────────────────────────────────
from sentinel.decision_engine import DecisionTier, THREAT_CONSERVATISM
from sentinel.simulator       import run_scenario, TickState, choose_holding_action
from sentinel.safety_gate     import (
    validate_command, ValidationResult,
    blackout_survival_loop, is_action_safe,
)
from sentinel.reasoning       import generate_reasoning, make_block_report
from sentinel.decision_manager import DecisionManager, MissionState
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Lookup tables
# ─────────────────────────────────────────────────────────────────────────────
TC = {"GREEN": "#16a34a", "YELLOW": "#ca8a04", "RED": "#dc2626"}
BG = {"GREEN": "#dcfce7", "YELLOW": "#fef9c3", "RED": "#fee2e2"}
IC = {"GREEN": "🟢",      "YELLOW": "🟡",      "RED": "🔴"}
PS = {"EMERGENCY": "🚨", "AUTONOMOUS": "🤖", "EARTH": "📡", "IDLE": "💤"}

TIER_ACTION = {
    "GREEN":  "Waiting for Earth response.",
    "YELLOW": "Holding — safe action active, Earth notified.",
    "RED":    "Acting autonomously NOW — notifying Earth after.",
}

SCENARIO_LABELS = {
    "cliff_edge":       "🪨  Cliff Edge",
    "dust_storm":       "🌪️  Dust Storm",
    "battery_critical": "🔋  Battery Critical",
    "rockfall":         "⛰️  Rockfall",
    "comms_blackout":   "📡  Comms Blackout",
}

SENSOR_UNITS = {
    "distance_m": "m", "drift_speed_ms": "m/s",
    "wind_speed_ms": "m/s", "dust_density_gcm3": "g/cm³", "optical_depth": "",
    "charge_pct": "%", "draw_pct_per_tick": "%/tick",
    "seismic_g": "g", "debris_dist_m": "m", "debris_speed_ms": "m/s",
    "relay_elevation_deg": "°", "effective_descent_rate": "°/tick",
}

ALL_COMMANDS = [
    "move_forward", "continue_heading", "increase_speed",
    "move_backward", "reverse", "turn_left", "turn_right",
    "hold_position", "stop",
    "transmit_data", "deploy_antenna", "run_diagnostics",
    "enable_heaters", "activate_drill",
]

# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _badge(tier_val: str, holding: str | None = None, source: str = "") -> str:
    c, b, i = TC[tier_val], BG[tier_val], IC[tier_val]
    src_html = (
        f'<br><span style="font-size:0.72rem;color:{c};opacity:0.8;">'
        f'{PS.get(source,"")} {source}</span>' if source else ""
    )
    hold_html = ""
    if holding:
        hico = "🧲" if holding == "hold_in_place" else "🔄"
        hold_html = (
            f'<br><span style="font-size:0.78rem;color:{c};background:rgba(0,0,0,0.06);'
            f'border-radius:4px;padding:2px 8px;">{hico} {holding.replace("_"," ")}</span>'
        )
    return (
        f'<div style="background:{b};border:2px solid {c};border-radius:12px;'
        f'padding:16px 24px;text-align:center;">'
        f'<span style="font-size:2.2rem;font-weight:800;color:{c};">{i} {tier_val}</span>'
        f'<br><span style="color:{c};font-size:0.82rem;">{TIER_ACTION[tier_val]}</span>'
        f'{hold_html}{src_html}</div>'
    )

def _ratio_bar(ratio: float) -> str:
    pct = min(ratio / 3.0, 1.0) * 100
    tv  = "GREEN" if ratio > 2 else ("YELLOW" if ratio > 1 else "RED")
    return (
        f'<div style="background:#e5e7eb;border-radius:6px;height:12px;width:100%;">'
        f'<div style="background:{TC[tv]};width:{pct:.1f}%;height:12px;border-radius:6px;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#6b7280;margin-top:2px;">'
        f'<span>0 RED</span><span>1×RTT</span><span>2×RTT</span><span>3×</span></div>'
    )

def _card(title: str, body: str, color: str = "#3b82d4") -> str:
    return (
        f'<div style="border-left:4px solid {color};background:#f7f8fa;'
        f'border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;">'
        f'<div style="font-size:0.72rem;color:#57606a;margin-bottom:4px;">{title}</div>'
        f'<div style="color:#1f2328;">{body}</div></div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Cached AI reasoning wrapper
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_reasoning(threat_type, sensors_repr, tth, rtt, ratio, tier_val, action):
    return generate_reasoning({
        "threat_type": threat_type, "sensors": sensors_repr,
        "time_to_harm_s": tth, "round_trip_s": rtt,
        "ratio": ratio, "tier": tier_val, "action": action,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _reset():
    st.session_state.tick_ptr    = -1
    st.session_state.ticks_data  = []      # list[TickState] from run_scenario
    st.session_state.ms_history  = []      # list[MissionState] from DecisionManager
    st.session_state.history_rows= []      # list[dict] for the table
    st.session_state.log_feed    = []
    st.session_state.cmd_log     = []
    st.session_state.running     = False
    st.session_state.dm          = None    # DecisionManager instance

if "tick_ptr" not in st.session_state:
    _reset()
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
        "One-way comm delay (s)", 240, 1440, 780, 60,
        help="Mars: ~240 s (near) → ~1440 s (far). Default ≈ 13 min.",
    )
    n_ticks     = st.slider("Ticks to simulate", 10, 35, 22)
    tick_delay  = st.slider("Delay between ticks (s)", 0.3, 3.0, 0.8, 0.1)
    ai_on       = st.toggle("AI reasoning (watsonx.ai)", value=True)
    anomaly_on  = st.toggle("Anomaly detection (IsolationForest)", value=True,
                             help="Run classify_sensor_pattern() on every tick via DecisionManager.")
    timeout_ticks = st.slider(
        "Earth timeout (ticks)", 1, 20, 4,
        help="If rover waits in GREEN for this many ticks without Earth response, "
             "tier is re-evaluated (may escalate to YELLOW/RED).",
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        start_btn = st.button("▶ Run", type="primary", use_container_width=True)
    with c2:
        reset_btn = st.button("↺ Reset", use_container_width=True)
    st.divider()
    st.caption(f"RTT = **{comm_delay*2:,} s**")
    st.caption("GREEN > 2× RTT  |  YELLOW 1–2×  |  RED ≤ 1×")

# ─────────────────────────────────────────────────────────────────────────────
# Button handlers
# ─────────────────────────────────────────────────────────────────────────────
if reset_btn:
    _reset()
    st.rerun()

if start_btn:
    _reset()
    st.session_state.ticks_data = list(
        run_scenario(scenario, ticks=n_ticks, comm_delay_s=comm_delay)
    )
    st.session_state.dm = DecisionManager(
        comm_delay_s        = comm_delay,
        tick_duration_s     = 30,
        earth_timeout_ticks = timeout_ticks,
        anomaly_window_size = 10,
        use_anomaly_model   = anomaly_on,
    )
    st.session_state.running  = True
    st.session_state.tick_ptr = 0

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0;'>🛰️ Sentinel Protocol</h1>"
    "<p style='color:#57606a;margin-top:4px;'>"
    "Autonomous Decision-Time-Budget Engine — Live Mission Dashboard</p>",
    unsafe_allow_html=True,
)
st.divider()

if st.session_state.tick_ptr == -1:
    st.info("Select a threat scenario in the sidebar and press **▶ Run** to start.", icon="ℹ️")

    # ── Static workflow explainer shown on idle screen ──────────────────────
    st.markdown("### System Workflow")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            _card("Step 1 — Sensor Ingestion",
                  "Raw sensor readings are pushed into a rolling window and scored "
                  "by the IsolationForest anomaly model each tick.",
                  "#3b82d4"),
            unsafe_allow_html=True,
        )
        st.markdown(
            _card("Step 4 — Earth Timeout (M1 fix)",
                  "If no Earth response arrives within the configured timeout, "
                  "the tier is re-evaluated against current (worsened) sensors. "
                  "GREEN may escalate to YELLOW or RED.",
                  "#f59e0b"),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            _card("Step 2 — Anomaly Detection (M2 fix)",
                  "IsolationForest classifies each window. Matched threats are routed "
                  "to classify_threat() for a real TTH-based tier. Unclassified anomalies "
                  "default to YELLOW + all-risky-actions-blocked.",
                  "#3b82d4"),
            unsafe_allow_html=True,
        )
        st.markdown(
            _card("Step 5 — Priority Arbitration (M5 fix)",
                  "EMERGENCY (RED) overrides everything. AUTONOMOUS (YELLOW) "
                  "interrupts unsafe Earth commands. EARTH (GREEN) proceeds only "
                  "when sensors remain safe on every re-check.",
                  "#7c5cd8"),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            _card("Step 3 — Tier Classification",
                  "classify_threat(threat, TTH, comm_delay) → GREEN / YELLOW / RED "
                  "using per-threat conservatism multipliers (0.70–1.00).",
                  "#3b82d4"),
            unsafe_allow_html=True,
        )
        st.markdown(
            _card("Step 6 — Command Interrupt (M3 fix)",
                  "While an approved Earth command is executing, sensors are re-checked "
                  "every tick. If conditions change, is_action_safe() blocks the command "
                  "mid-execution before any movement occurs.",
                  "#ef4444"),
            unsafe_allow_html=True,
        )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Advance one tick
# ─────────────────────────────────────────────────────────────────────────────
all_ticks: list[TickState]  = st.session_state.ticks_data
ptr:       int               = st.session_state.tick_ptr
dm:        DecisionManager   = st.session_state.dm

if st.session_state.running and ptr < len(all_ticks):
    ts: TickState = all_ticks[ptr]

    # ── Run DecisionManager tick (wires anomaly + timeout + command interrupt) ──
    ms: MissionState = dm.tick(ts.sensors)
    st.session_state.ms_history.append(ms)

    # Prefer DecisionManager tier (full arbitration) over raw simulator tier
    tier_val = ms.tier.value

    conservatism = THREAT_CONSERVATISM[scenario]
    rtt          = comm_delay * 2
    adj_tth      = ts.time_to_harm_s * conservatism
    ratio        = adj_tth / rtt if rtt > 0 else float("inf")

    # AI reasoning
    if ai_on:
        ai_text = _cached_reasoning(
            threat_type  = scenario,
            sensors_repr = str(ts.sensors),
            tth          = ts.time_to_harm_s,
            rtt          = rtt,
            ratio        = ratio,
            tier_val     = tier_val,
            action       = TIER_ACTION[tier_val],
        )
    else:
        ai_text = "(AI reasoning disabled)"

    st.session_state.log_feed.append({
        "tick": ptr, "tier": tier_val, "text": ai_text,
        "source": ms.priority_source,
        "notes": ms.notes,
        "timeout": ms.earth_timeout,
        "interrupted": ms.cmd_interrupted,
    })

    ha_val = ms.holding_action or ts.holding_action or ""
    st.session_state.history_rows.append({
        "Tick":           ptr,
        "Tier":           f"{IC[tier_val]} {tier_val}",
        "Source":         ms.priority_source,
        "Holding Action": ha_val,
        "TTH (s)":        ts.time_to_harm_s,
        "Adj Ratio":      round(ratio, 3),
        "Timeout":        "⏰" if ms.earth_timeout else "",
        "Interrupted":    "⚡" if ms.cmd_interrupted else "",
        **ts.sensors,
    })

    st.session_state.tick_ptr += 1
    if st.session_state.tick_ptr >= len(all_ticks):
        st.session_state.running = False

# ─────────────────────────────────────────────────────────────────────────────
# Guard: nothing rendered yet
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.history_rows:
    st.info("Starting simulation…")
    time.sleep(0.3)
    st.rerun()

rows      = st.session_state.history_rows
last_row  = rows[-1]
last_ts   = all_ticks[last_row["Tick"]]
last_ms   = st.session_state.ms_history[-1] if st.session_state.ms_history else None
last_tv   = last_row["Tier"].split(" ")[-1]   # strip emoji

conservatism = THREAT_CONSERVATISM[scenario]
rtt          = comm_delay * 2
adj_tth      = last_ts.time_to_harm_s * conservatism
ratio        = adj_tth / rtt if rtt > 0 else float("inf")
n_done       = len(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — progress + tier badge + metrics
# ─────────────────────────────────────────────────────────────────────────────
prog_col, badge_col, tth_col, ratio_col, wait_col = st.columns([2, 2, 1, 1, 1])

with prog_col:
    lbl = "▶ running…" if st.session_state.running else "✅ complete"
    st.markdown(f"**Tick {n_done}/{len(all_ticks)}** — `{scenario}` — {lbl}")
    st.progress(n_done / len(all_ticks))
    if last_ms and last_ms.earth_timeout:
        st.warning("⏰ Earth timeout — tier escalated from sensor re-evaluation", icon="⏰")
    if last_ms and last_ms.cmd_interrupted:
        st.error("⚡ In-flight Earth command interrupted by safety gate", icon="⚡")

with badge_col:
    src = last_ms.priority_source if last_ms else "IDLE"
    st.markdown(
        _badge(last_tv, last_ms.holding_action if last_ms else None, src),
        unsafe_allow_html=True,
    )

with tth_col:
    prev = all_ticks[last_row["Tick"] - 1].time_to_harm_s if last_row["Tick"] > 0 else None
    delta = f"{last_ts.time_to_harm_s - prev:+.0f} s" if prev is not None else None
    st.metric("Time-to-Harm", f"{last_ts.time_to_harm_s:,.0f} s", delta=delta)

with ratio_col:
    st.metric("Adj Ratio", f"{ratio:.3f}", help="> 2 = GREEN | 1–2 = YELLOW | ≤ 1 = RED")

with wait_col:
    wt = last_ms.ticks_waiting if last_ms else 0
    st.metric("Waiting ticks", wt,
              help=f"Ticks in GREEN without Earth response (timeout={timeout_ticks})")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — sensors | ratio bar | AI log
# ─────────────────────────────────────────────────────────────────────────────
sensor_col, log_col = st.columns([1, 2])

with sensor_col:
    st.markdown("#### 📡 Sensor Readings")
    for k, v in last_ts.sensors.items():
        unit  = SENSOR_UNITS.get(k, "")
        label = k.replace("_", " ").title()
        st.metric(label, f"{v} {unit}".strip())
    st.markdown("#### Decision Ratio")
    st.markdown(_ratio_bar(ratio), unsafe_allow_html=True)
    st.caption(f"RTT={rtt:,}s  ·  conservatism={conservatism:.0%}")

    # ── Anomaly panel ────────────────────────────────────────────────────────
    if anomaly_on and last_ms and last_ms.anomaly:
        a = last_ms.anomaly
        st.markdown("#### 🔬 Anomaly Detection")
        ac = TC.get(a.tier.value, "#57606a")
        ab = BG.get(a.tier.value, "#f7f8fa")
        flag = "🔴 ANOMALY" if a.is_anomaly else "🟢 Normal"
        st.markdown(
            f'<div style="background:{ab};border:1px solid {ac};border-radius:8px;'
            f'padding:10px 14px;font-size:0.84rem;">'
            f'<b style="color:{ac};">{flag}</b><br>'
            f'Score: {a.anomaly_score:.3f} &nbsp;|&nbsp; '
            f'Type: <code>{a.threat_type or "—"}</code><br>'
            f'{a.label}</div>',
            unsafe_allow_html=True,
        )

with log_col:
    st.markdown("#### 🤖 AI Mission Log")
    feed = st.session_state.log_feed
    if not feed:
        st.caption("No entries yet.")
    else:
        for entry in reversed(feed):
            t      = entry["tier"]
            c, b   = TC[t], BG[t]
            badges = ""
            if entry.get("timeout"):
                badges += ' <span style="background:#fef3c7;color:#92400e;border-radius:3px;padding:1px 5px;font-size:0.7rem;">⏰ timeout</span>'
            if entry.get("interrupted"):
                badges += ' <span style="background:#fee2e2;color:#991b1b;border-radius:3px;padding:1px 5px;font-size:0.7rem;">⚡ interrupted</span>'
            notes_html = ""
            for note in entry.get("notes", []):
                notes_html += f'<div style="font-size:0.72rem;color:#57606a;margin-top:2px;">↳ {note}</div>'
            st.markdown(
                f'<div style="border-left:4px solid {c};background:{b};'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:8px;">'
                f'<div style="font-size:0.72rem;color:#57606a;">'
                f'Tick {entry["tick"]} · {IC[t]} {t} · {PS.get(entry["source"],"")} {entry["source"]}'
                f'{badges}</div>'
                f'<div style="color:#1f2328;margin-top:4px;">{entry["text"]}</div>'
                f'{notes_html}</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 3 — Tick history table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📋 Tick History")
df = pd.DataFrame(rows)

def _color_tier(val: str):
    if "GREEN"  in str(val): return "background-color:#dcfce7;color:#16a34a;font-weight:600"
    if "YELLOW" in str(val): return "background-color:#fef9c3;color:#ca8a04;font-weight:600"
    if "RED"    in str(val): return "background-color:#fee2e2;color:#dc2626;font-weight:600"
    return ""

base_cols = ["Tick", "Tier", "Source", "Holding Action", "TTH (s)", "Adj Ratio", "Timeout", "Interrupted"]
sensor_cols = [c for c in df.columns if c not in base_cols]
display_cols = base_cols + sensor_cols

styled = (
    df[display_cols].style
    .map(_color_tier, subset=["Tier"])
    .format({"TTH (s)": "{:,.0f}", "Adj Ratio": "{:.3f}"}, na_rep="—")
)
st.dataframe(styled, use_container_width=True, height=min(60 + len(df) * 35, 420))

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 4 — Earth Command Terminal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📡 Earth Command Terminal")
st.caption(
    "Simulate a command from Earth. "
    "Commands are validated through the full safety gate against the current "
    "sensor state **and all active threats simultaneously**. "
    "While the simulation is running, queued commands are re-validated live "
    "each tick — an approved command can be interrupted if sensors worsen (M3)."
)

cmd_col, send_col = st.columns([3, 1])
with cmd_col:
    selected_cmd = st.selectbox(
        "Command from Earth", ALL_COMMANDS, key="cmd_select",
        label_visibility="collapsed",
    )
with send_col:
    send_btn = st.button("📤 Send to rover", type="primary", use_container_width=True)

if send_btn and last_ms is not None:
    # Pass ALL active threats (M4 fix — multi-threat validation)
    active_threats = last_ms.active_threats or [scenario]
    result: ValidationResult = validate_command(
        command          = selected_cmd,
        sensor_state     = last_ts.sensors,
        threat_type      = active_threats,
        comm_delay_s     = comm_delay,
        _block_report_fn = make_block_report,
    )
    if st.session_state.running:
        # Queue for live re-validation during execution
        dm.queue_earth_command(selected_cmd)

    st.session_state.cmd_log.insert(0, {
        "tick":    n_done,
        "tier":    last_tv,
        "cmd":     selected_cmd,
        "verdict": result.verdict,
        "reason":  result.reason,
        "report":  result.earth_report,
        "threats": active_threats,
    })

# Latest command result
if st.session_state.cmd_log:
    latest  = st.session_state.cmd_log[0]
    verdict = latest["verdict"]
    threat_label = ", ".join(latest.get("threats", []))

    if verdict == "BLOCKED":
        st.markdown(
            f'<div style="border:2px solid #dc2626;background:#fee2e2;'
            f'border-radius:10px;padding:16px 20px;margin-top:8px;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#dc2626;margin-bottom:6px;">'
            f'🚫 COMMAND BLOCKED — Tick {latest["tick"]} · {IC.get(latest["tier"],"")} {latest["tier"]}</div>'
            f'<div style="margin-bottom:6px;"><b>Command:</b> <code>{latest["cmd"]}</code></div>'
            f'<div style="margin-bottom:6px;"><b>Active threats:</b> <code>{threat_label}</code></div>'
            f'<div style="color:#7f1d1d;font-size:0.88rem;margin-bottom:10px;">'
            f'<b>Conflict:</b> {latest["reason"]}</div>'
            f'<div style="border-top:1px solid #fca5a5;padding-top:10px;">'
            f'<b>AI report to Earth</b> '
            f'<span style="font-size:0.72rem;color:#57606a;">(ibm/granite-4-h-small)</span><br>'
            f'<em>"{latest["report"]}"</em></div></div>',
            unsafe_allow_html=True,
        )
    else:
        queued = " — <em>queued for live re-validation during execution</em>" if st.session_state.running else ""
        st.markdown(
            f'<div style="border:2px solid #16a34a;background:#dcfce7;'
            f'border-radius:10px;padding:16px 20px;margin-top:8px;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#16a34a;margin-bottom:6px;">'
            f'✅ COMMAND APPROVED — Tick {latest["tick"]} · {IC.get(latest["tier"],"")} {latest["tier"]}</div>'
            f'<div><b>Command:</b> <code>{latest["cmd"]}</code> — passed through to rover.{queued}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if len(st.session_state.cmd_log) > 1:
        with st.expander(f"Command history ({len(st.session_state.cmd_log)} sent)"):
            for e in st.session_state.cmd_log:
                vc = "#dc2626" if e["verdict"] == "BLOCKED" else "#16a34a"
                vi = "🚫" if e["verdict"] == "BLOCKED" else "✅"
                st.markdown(
                    f'<div style="border-left:3px solid {vc};padding:6px 12px;'
                    f'margin-bottom:6px;background:#f7f8fa;border-radius:4px;">'
                    f'<span style="font-size:0.78rem;color:#57606a;">Tick {e["tick"]} · {e["tier"]}</span>'
                    f' &nbsp;{vi} <b style="color:{vc};">{e["verdict"]}</b>'
                    f' &nbsp;<code>{e["cmd"]}</code>'
                    + (f'<br><span style="font-size:0.75rem;color:#57606a;">{e["reason"]}</span>'
                       if e["reason"] else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 5 — Blackout Survival Loop tester
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🌑 Blackout Survival Loop — Interactive Test", expanded=False):
    st.caption(
        "Simulate a total comms blackout starting from the current sensor state. "
        "Every action is re-validated through is_action_safe() before execution."
    )
    bt_col, bb_col = st.columns([3, 1])
    with bt_col:
        bt_thresh = st.slider(
            "Battery rescue threshold (%)", 5, 30, 12,
            key="bt_thresh",
            help="Navigate to sunlight when charge drops below this level.",
        )
    with bb_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_blackout = st.button("▶ Run blackout loop", use_container_width=True, key="run_bl")

    if run_blackout:
        steps = list(blackout_survival_loop(
            sensor_state          = last_ts.sensors,
            comm_delay_s          = comm_delay,
            battery_rescue_thresh = bt_thresh,
            max_wait_steps        = 6,
        ))
        for step in steps:
            sc = TC.get(step.safety.safe and "GREEN" or "RED", "#dc2626")
            bg = BG.get(step.safety.safe and "GREEN" or "RED", "#fee2e2")
            ok = "✅" if step.executed else ("🚫" if not step.safety.safe else "⏭️")
            st.markdown(
                f'<div style="border-left:4px solid {sc};background:{bg};'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:6px;">'
                f'<b style="color:{sc};">[{step.phase}]</b> &nbsp;{ok} &nbsp;'
                f'<code>{step.proposed}</code><br>'
                f'<span style="font-size:0.82rem;color:#1f2328;">{step.note}</span>'
                + (f'<br><span style="font-size:0.75rem;color:#7f1d1d;">⚠ {step.safety.reason}</span>'
                   if not step.safety.safe else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Row 6 — Manual safety gate probe
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍 Safety Gate — Manual Probe", expanded=False):
    st.caption(
        "Test any action against any combination of active threats and custom sensor values. "
        "This calls is_action_safe() directly — bypasses the scenario to let you probe edge cases."
    )
    pg_c1, pg_c2 = st.columns(2)
    with pg_c1:
        probe_action = st.selectbox("Action", ALL_COMMANDS + ["emergency_full_stop", "cut_motors"], key="probe_action")
        probe_threats = st.multiselect(
            "Active threats",
            ["cliff_edge","dust_storm","battery_critical","rockfall","comms_blackout","unclassified_anomaly"],
            default=[scenario],
            key="probe_threats",
        )
    with pg_c2:
        probe_dist    = st.number_input("distance_m",          0.0, 5000.0, 50.0,  key="pd")
        probe_charge  = st.number_input("charge_pct",          0.0,  100.0, 30.0,  key="pc")
        probe_wind    = st.number_input("wind_speed_ms",        0.0,   60.0,  5.0,  key="pw")
        probe_debris  = st.number_input("debris_dist_m",        0.0, 5000.0, 500.0, key="pdb")

    probe_btn = st.button("🔎 Check", key="probe_btn")
    if probe_btn:
        probe_sensors = {
            "distance_m":      probe_dist,
            "drift_speed_ms":  0.05,
            "charge_pct":      probe_charge,
            "draw_pct_per_tick": 0.5,
            "wind_speed_ms":   probe_wind,
            "optical_depth":   0.1,
            "debris_dist_m":   probe_debris,
            "debris_speed_ms": 1.0,
            "relay_elevation_deg": 30.0,
        }
        r = is_action_safe(probe_action, probe_sensors, probe_threats, comm_delay)
        col_r = TC["GREEN"] if r.safe else TC["RED"]
        bg_r  = BG["GREEN"] if r.safe else BG["RED"]
        label = "✅ SAFE" if r.safe else "🚫 BLOCKED"
        st.markdown(
            f'<div style="border:2px solid {col_r};background:{bg_r};'
            f'border-radius:8px;padding:14px 18px;">'
            f'<b style="color:{col_r};font-size:1.1rem;">{label}</b><br>'
            f'<b>Action:</b> <code>{probe_action}</code> &nbsp;|&nbsp; '
            f'<b>Threats:</b> <code>{", ".join(probe_threats) or "none"}</code>'
            + (f'<br><b>Blocked by:</b> <code>{r.blocked_by}</code>'
               f'<br><b>Reason:</b> {r.reason}' if not r.safe else "")
            + f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Schedule next tick — ALWAYS at very bottom after all UI is drawn
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(tick_delay)
    st.rerun()
