# Sentinel Protocol

**AI-powered autonomy engine for planetary field robots — deciding, in real time, when to act alone and when to wait for Earth.**

## Problem Statement

Planetary missions rely on rovers and field robots operating far from Earth, where communication delays range from several minutes (Mars) to complete blackouts during solar interference or terrain obstruction. In these gaps, a rover facing a sudden hazard — a cliff edge, a dust storm, a critical battery drop, falling debris, or a total loss of signal — cannot simply wait for human instruction if the threat will become irreversible before a response can arrive. Mission teams and the robots themselves need a way to know, moment to moment, whether a situation is safe to escalate to Earth or urgent enough to demand immediate autonomous action. Without this, missions risk either dangerous delays or reckless unsupervised decisions.

## Solution Description

Sentinel Protocol is an AI decision-support engine that continuously evaluates incoming hazard signals and classifies each situation into one of three response tiers — **Green** (safe to wait for Earth), **Yellow** (take a safe holding action while notifying Earth), or **Red** (act immediately, report afterward). It does this by comparing an estimated *time-to-harm* against the *time it would take to hear back from Earth*, using a transparent, explainable decision-time-budget calculation rather than an opaque model. Every decision Sentinel makes is logged with a plain-language explanation of its reasoning, so mission controllers can trust and audit its choices after the fact.

## Scope: Threat Scenarios (Step 1)

To keep the system focused and demonstrable, Sentinel is built around five representative hazard types, each defined by a detectable signal, an estimated window before the danger becomes irreversible, and a safe fallback action if there isn't time to wait for Earth.

| Threat | Signal (simulated sensor) | Time-to-Harm | Safe Action if No Time to Wait |
|---|---|---|---|
| **Cliff edge** | Distance-to-edge shrinking + forward velocity | Seconds (5–15s) | Stop immediately, reverse 2m |
| **Dust storm** | Wind speed + dust density rising | Minutes (2–10min) | Park, lower antenna, shield panels if possible |
| **Battery critical** | Battery % dropping + no sunlight | Minutes–hours | Reposition to sunlight, enter low-power mode |
| **Rockfall / debris** | Vibration sensor + camera obstruction spike | Seconds (3–10s) | Halt, shield cameras, wait it out |
| **Comms blackout** | Signal strength drops to zero (solar interference or terrain) | Unknown duration | Switch to full autonomy: prioritize battery, then safety, then position; queue all logs for Earth |

These five scenarios form the basis for the decision engine, the live scenario simulator, and the demo dashboard described in the sections below.

## AI Approach and Architecture

At the core of Sentinel Protocol is a **decision-time-budget engine**: for every detected hazard, the system estimates *time-to-harm* (how long until the situation becomes irreversible) and compares it against the *round-trip communication delay* to Earth. Rather than a single hard cutoff, the engine applies a **conservatism buffer** (a scenario-specific safety margin, e.g. 70–100% depending on hazard type) to the raw ratio of time-to-harm over round-trip delay, producing an adjusted ratio that determines the response tier:

- 🔴 **RED** — adjusted ratio is very low (harm is imminent relative to Earth's response time) → the rover acts autonomously immediately and notifies Earth afterward.
- 🟡 **YELLOW** — adjusted ratio is borderline (there's some time, but not comfortably enough) → the rover executes a safe holding action while notifying Earth in parallel.
- 🟢 **GREEN** — adjusted ratio is comfortably high (plenty of time relative to the round-trip delay) → the rover waits for Earth's response before acting.

This approach keeps every decision transparent and auditable — there is no black-box classifier, only a clear, explainable calculation that a mission controller could review and verify after the fact.

**Validated test scenarios** (round-trip comms delay: 1,560s):

| Scenario | Time-to-Harm | Adjusted Ratio | Decision |
|---|---|---|---|
| Cliff edge (4m away, closing at 0.02 m/s) | 200s | 0.103 | 🔴 RED |
| Dust storm (90 min out) | 5,400s | 3.115 | 🟢 GREEN |
| Battery critical (8%, 40 min to shutdown) | 2,400s | 1.462 | 🟡 YELLOW |
| Rockfall (imminent slope collapse) | 8s | 0.004 | 🔴 RED |
| Comms blackout (relay occultation in 35 min) | 2,100s | 1.346 | 🟡 YELLOW |

All five scenarios classified correctly against expected outcomes.

### AI Reasoning Layer (IBM watsonx.ai / Granite)

On top of the rule-based decision engine, Sentinel Protocol integrates **IBM watsonx.ai**, calling the `ibm/granite-4-h-small` foundation model via the `/ml/v1/text/chat` endpoint to generate natural-language mission-log explanations for every decision. Given a tick's full context (threat type, sensor readings, time-to-harm, comm delay, adjusted ratio, tier, and required action), the model produces a concise, professional log entry — written in the voice of a flight engineer — explaining exactly why the system chose to act autonomously or wait for Earth. This is the project's core AI component: it turns a transparent but silent decision engine into a system that can explain itself in plain language, which is essential for mission controllers to trust and audit autonomous actions after the fact.

**Example generated output** (Tick 9, cliff_edge scenario, YELLOW → RED transition):

> "Sentinel autonomously executed evasive maneuvers to avoid cliff edge, initiating immediate hazard mitigation protocol per RED-tier directive; Earth notification scheduled post-action completion."

## Live Dashboard

Sentinel Protocol includes a standalone **Streamlit dashboard** (`dashboard.py`) that visualizes a scenario run in real time: live sensor readings for the active threat, the current decision tier with clear color coding (green/yellow/red), a live-updating AI-generated mission-log reasoning feed, and a scrolling history of past ticks and decisions. Users select a threat scenario from a dropdown (cliff_edge, dust_storm, battery_critical, rockfall, comms_blackout) and watch it play out tick by tick, mirroring how Sentinel would behave on an actual mission.

## Setup and How to Run

**Requirements:** Python 3.12+, an IBM watsonx.ai account with an active project and API key.

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your watsonx credentials as environment variables (do **not** hardcode them):
   - `WATSONX_API_KEY`
   - `WATSONX_PROJECT_ID`
4. **Decision engine & simulator:** open `sentinel_decision_engine.ipynb` in JupyterLab to explore the core classification logic and run scenario simulations.
5. **Live dashboard:** run `streamlit run dashboard.py` and open the local URL it provides in your browser.

## Challenge Theme

Advance Space Exploration with AI — August Challenge

## How IBM Bob Was Used

IBM Bob was used across the full build of Sentinel Protocol's backend in JupyterLab: designing and implementing the `classify_threat()` decision engine and its `Threat` dataclass, building the tick-by-tick scenario simulator that feeds realistic escalating sensor data into the engine, and integrating IBM watsonx.ai's `granite-4-h-small` model to generate the natural-language reasoning layer. Bob proposed the LLM prompt template and API integration approach, surfaced configuration decisions (model selection, endpoint, project ID) for explicit approval rather than assuming defaults, and iterated on the watsonx integration after an initial deprecated-endpoint issue was identified and corrected.