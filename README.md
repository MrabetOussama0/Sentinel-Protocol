<p align="center">
  <img src="assets/logo.svg" alt="Sentinel Protocol logo" width="500"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/Built%20with-IBM%20Bob-0F6E56" alt="Built with IBM Bob"/>
  <img src="https://img.shields.io/badge/Powered%20by-IBM%20watsonx.ai-534AB7" alt="Powered by watsonx.ai"/>
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License"/>
</p>

# Sentinel Protocol

**AI-powered autonomy engine for planetary field robots — deciding, in real time, when to act alone and when to wait for Earth.**

### Contents
- [Problem Statement](#problem-statement)
- [Solution Description](#solution-description)
- [System Architecture](#system-architecture)
- [Scope: Threat Scenarios](#scope-threat-scenarios-step-1)
- [AI Approach and Architecture](#ai-approach-and-architecture)
- [Live Dashboard](#live-dashboard)
- [Setup and How to Run](#setup-and-how-to-run)
- [Challenge Theme](#challenge-theme)
- [How IBM Bob Was Used](#how-ibm-bob-was-used)



## Problem Statement

Planetary missions rely on rovers and field robots operating far from Earth, where communication delays range from several minutes (Mars) to complete blackouts during solar interference or terrain obstruction. In these gaps, a rover facing a sudden hazard — a cliff edge, a dust storm, a critical battery drop, falling debris, or a total loss of signal — cannot simply wait for human instruction if the threat will become irreversible before a response can arrive. Mission teams and the robots themselves need a way to know, moment to moment, whether a situation is safe to escalate to Earth or urgent enough to demand immediate autonomous action. Without this, missions risk either dangerous delays or reckless unsupervised decisions.

This is not a hypothetical risk. NASA's twin Mars Exploration Rovers, Spirit and Opportunity, both suffered mission-ending failures rooted in exactly this gap between hazard and response time. In 2009, Spirit drove into a patch of soft sand it could not recognize as dangerous in time, became permanently stuck, and was unable to reposition itself to keep its solar panels angled toward the sun — it lost power over the following Martian winter and never made contact again. In 2018, a planet-wide dust storm cut off Opportunity's sunlight for weeks; the rover entered a low-power fault and lost communication with Earth, ending its 14-year mission. In both cases, a system capable of recognizing the danger early and independently choosing a self-preserving action — such as halting before entering unstable terrain, or proactively returning to a known safe position with reliable sunlight before a storm fully cut off power and communication — could plausibly have changed the outcome.

Sentinel Protocol is built around this exact gap: giving a field robot the judgment to recognize when a threat cannot wait for Earth, and the ability to choose a genuine self-rescue action — not just stopping in place, but actively repositioning toward safety, entering a low-power holding state, and queuing a full status report to send the moment contact with Earth is restored.

## Solution Description

Sentinel Protocol is an AI decision-support engine that continuously evaluates incoming hazard signals and classifies each situation into one of three response tiers — **Green** (safe to wait for Earth), **Yellow** (take a safe holding action while notifying Earth), or **Red** (act immediately, report afterward). It does this by comparing an estimated *time-to-harm* against the *time it would take to hear back from Earth*, using a transparent, explainable decision-time-budget calculation rather than an opaque model. Every decision Sentinel makes is logged with a plain-language explanation of its reasoning, so mission controllers can trust and audit its choices after the fact.

## System Architecture

Sentinel Protocol runs as the onboard intelligence of a rover carrying both its scientific instruments (soil analysis, atmospheric sensors, and similar equipment) and the AI decision-making system itself. Rather than reacting passively to instructions, the rover's AI sits between Earth and the rover's own actuators, playing two distinct safety roles:

1. **Autonomous hazard response** — when the rover detects an emerging danger (a cliff, a storm, falling debris, a battery crisis, or a loss of contact with Earth), it uses the decision-time-budget engine to decide whether there's time to escalate to Earth or whether it must act immediately.

2. **Command validation** — before executing an incoming order from Earth, the rover's AI evaluates whether carrying out that command as given could lead to harm given what it currently observes. For example, if Earth instructs the rover to continue moving forward but its sensors show a cliff edge ahead that Earth's last data update didn't capture, the AI intervenes: it halts the command, holds position, and reports the conflict back to Earth rather than executing an order that could destroy the mission. If no conflict is detected, the command is passed through and executed normally. This means Sentinel isn't only protecting against environmental hazards — it also acts as a safeguard against commands based on outdated or incomplete information reaching Earth's operators, without ever overriding Earth's authority outright.

Both roles are implementations of a single underlying principle, detailed further below: no action — whether ordered by Earth or generated by the rover itself — executes without first passing a safety check.

<p align="center">
  <img src="assets/architecture-diagram.svg" alt="Sentinel Protocol architecture diagram: sensor inputs and Earth commands both flow through the decision-time-budget engine and universal safety gate, producing a GREEN, YELLOW, or RED response, with a blackout survival loop and watsonx AI reasoning layer feeding the live dashboard" width="680"/>
</p>

## Scope: Threat Scenarios (Step 1)

To keep the system focused and demonstrable, Sentinel is built around five representative hazard types, each defined by a detectable signal, an estimated window before the danger becomes irreversible, and a safe fallback action if there isn't time to wait for Earth.

| Threat | Signal (simulated sensor) | Time-to-Harm | Safe Action if No Time to Wait |
|---|---|---|---|
| **Cliff edge** | Distance-to-edge shrinking + forward velocity | Seconds (5–15s) | Stop immediately, reverse 2m |
| **Dust storm** | Wind speed + dust density rising | Minutes (2–10min) | Park, lower antenna, shield panels if possible |
| **Battery critical** | Battery % dropping + no sunlight | Minutes–hours | Reposition to sunlight, enter low-power mode |
| **Rockfall / debris** | Vibration sensor + camera obstruction spike | Seconds (3–10s) | Halt, shield cameras, wait it out |
| **Comms blackout** | Signal strength drops to zero (solar interference or terrain) | Unknown duration | Switch to full autonomy: prioritize battery, then safety, then position; queue all logs for Earth |
| **Unclassified anomaly** ¹ | IsolationForest flags a sensor pattern that doesn't match any known threat | Unknown | Block all non-hold actions; only `stop` / `hold` permitted pending Earth confirmation |

¹ Not a sensor-driven threat type — raised by the anomaly detection layer when a reading is flagged anomalous but doesn't match any of the five known signatures. Handled by the safety gate's unconditional conservative default rather than the decision-time-budget engine.

These five scenarios form the basis for the decision engine, the live scenario simulator, and the demo dashboard described in the sections below.

## AI Approach and Architecture

At the core of Sentinel Protocol is a **decision-time-budget engine**: for every detected hazard, the system estimates *time-to-harm* (how long until the situation becomes irreversible) and compares it against the *round-trip communication delay* to Earth. Rather than a single hard cutoff, the engine applies a **conservatism buffer** (a scenario-specific safety margin, e.g. 70–100% depending on hazard type) to the raw ratio of time-to-harm over round-trip delay, producing an adjusted ratio that determines the response tier:

- 🔴 **RED** — adjusted ratio is very low (harm is imminent relative to Earth's response time) → the rover acts autonomously immediately and notifies Earth afterward.
- 🟡 **YELLOW** — adjusted ratio is borderline (there's some time, but not comfortably enough) → the rover executes a safe holding action — which may mean stopping in place, or, when time allows, actively repositioning toward a known safe location (e.g. a spot with reliable sunlight or a docking/charging station) — while notifying Earth in parallel.
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

### Command Validation

Beyond reacting to environmental hazards, Sentinel applies the same decision-time-budget engine to a second use case: validating commands *coming from Earth* before executing them. Every incoming instruction is checked with `validate_command()` against the current sensor state — if executing the command would run the rover into an active hazard (for example, Earth instructing "move forward" while a cliff edge is still within the danger window, potentially based on outdated data), the command is **blocked**, the rover holds position, and an AI-generated explanation is sent back to Earth instead of the order being blindly obeyed. If no conflict exists, the command is approved and passed through unchanged. Sentinel never overrides Earth's authority outright — it only refuses to execute an order it can verify is unsafe given what it currently observes, and always reports why.

### Universal Safety Gate and Blackout Survival Loop

Command validation and autonomous hazard response are both implementations of a single underlying principle: **no action — whether ordered by Earth or generated by the rover itself — executes without first passing a safety check.** This is implemented as one reusable gate, `is_action_safe()`, used by every part of the system.

This gate becomes critical during a **total communications blackout**, when Earth cannot be consulted at all. In that situation — for instance, comms are lost while the rover is near a cliff edge — Sentinel runs a continuous survival loop rather than a single reaction:

1. **Check if stopping is safe.** If holding in place passes the safety gate, the rover stops and holds, periodically checking for Earth contact to resume.
2. **If stopping is not safe** (e.g. already too close to an edge), the rover proposes a small corrective move — such as reversing a short distance — and that proposed move is itself validated through the same safety gate before it is executed. Nothing moves on an assumption; every step is checked.
3. **While holding and waiting for Earth to return**, Sentinel continues monitoring other systems. If a new problem emerges mid-wait — for example, battery drops to critical — it proposes navigating to the nearest known safe, sunlit location, and that path is again validated through the safety gate before the rover moves.

This means a total loss of contact does not leave the rover paralyzed or blindly reactive — it falls back to a single, consistent decision-making loop that treats every action, self-generated or externally ordered, with the same scrutiny.

### Handling Unrecognized Threats

Not every hazard the rover encounters will match one of its five known threat types. For a genuinely **unclassified anomaly** — a sensor pattern that doesn't fit any known category — Sentinel does not guess at what's safe. Rather than evaluating sensor values against thresholds it doesn't have, the safety gate applies an **unconditional block**: every action beyond simple holding (movement, transmission, antenna deployment, and other higher-risk commands) is blocked outright, and only hold-type actions (`stop`, `hold`, `hold_in_place`, `emergency_full_stop`) are permitted, pending clarification from Earth. This was a deliberate design correction made after testing revealed the opposite default — passing unclassified threats through as safe — was the wrong failure mode: when the nature of a hazard is unknown, the system defaults to maximum caution rather than assuming safety.

### Anomaly Detection (NASA SMAP/MSL)

Sentinel's five threat types cover known, well-defined hazard categories — but not every real-world sensor pattern will fit a category the system was explicitly built to recognize. To handle this, Sentinel includes an anomaly detection layer trained and evaluated on NASA's public **SMAP/MSL Anomaly Detection Dataset**, which contains expert-labeled telemetry from the Soil Moisture Active Passive satellite and the Mars Science Laboratory (Curiosity) rover — 517,764 timesteps across 82 telemetry channels, 12.5% of which are labeled anomalous.

A scikit-learn `IsolationForest` model is trained on channel-level statistical features (anomaly fraction, window length, timing patterns) derived from this data, then evaluated on a held-out set of channels it never saw during training:

| Metric | Result |
|---|---|
| Accuracy | 0.691 |
| Precision (anomalous) | **1.000** |
| Recall (anomalous) | 0.370 |
| F1 (anomalous) | 0.541 |

The model achieves **perfect precision — zero false positives** — meaning it never misclassifies a genuinely normal channel as anomalous, at the cost of a more conservative recall that only flags the clearest anomalies. This is a deliberate and appropriate trade-off for a rover safety system: a false alarm wastes time and erodes trust in the system, but a missed anomaly is caught downstream by Sentinel's other layers (the five known threat types and the unclassified-anomaly safety default below) — so the anomaly layer is tuned to be confident, not exhaustive.

At runtime, incoming sensor readings are scored against this model. If a reading is flagged anomalous and matches one of Sentinel's known threat signatures (e.g. rockfall, cliff_edge), it's routed into the standard `classify_threat()` engine as usual. If it's flagged anomalous but doesn't match any known threat, it is classified as an **unclassified anomaly** and handled by the safety gate's conservative default described above — every action beyond holding in place is blocked until Earth can confirm what the anomaly actually is.

## Live Dashboard


Sentinel Protocol includes a standalone **Streamlit dashboard** (`dashboard.py`) that visualizes a scenario run in real time: live sensor readings for the active threat, the current decision tier with clear color coding (green/yellow/red), a live-updating AI-generated mission-log reasoning feed, and a scrolling history of past ticks and decisions. Users select a threat scenario from a dropdown (cliff_edge, dust_storm, battery_critical, rockfall, comms_blackout) and watch it play out tick by tick, mirroring how Sentinel would behave on an actual mission.

<p align="center">
  <img src="assets/dashboard-screenshot.png" alt="Sentinel Protocol dashboard showing live sensor readings, a color-coded decision tier, and an AI-generated reasoning feed" width="680"/>
</p>

## Setup and How to Run

**Requirements:** Python 3.12+, an IBM watsonx.ai account with an active project and API key.

**Project structure:**
```
sentinel-protocol/
├── sentinel/                    # Core logic package
│   ├── decision_engine.py       # classify_threat(), Threat dataclass, THREAT_CONSERVATISM
│   ├── safety_gate.py           # is_action_safe(), validate_command(), blackout_survival_loop()
│   ├── reasoning.py             # generate_reasoning() — watsonx / Granite integration
│   ├── simulator.py             # run_scenario() tick-by-tick scenario generator
│   └── anomaly.py               # classify_sensor_pattern() — IsolationForest anomaly model
├── data/
│   ├── labeled_anomalies.csv    # NASA SMAP/MSL anomaly labels (82 channels, 517,764 timesteps)
│   └── anomaly_model.joblib     # Pre-trained IsolationForest pipeline — committed to repo,
│                                #   no re-training needed; loaded lazily on first import
├── notebooks/
│   └── sentinel_analysis.ipynb  # Demonstration and validation of the sentinel package
├── dashboard.py                 # Streamlit live dashboard
├── .env                         # Your watsonx credentials — NOT committed (see step 3 below)
├── requirements.txt
└── assets/                      # Logo, diagrams, dashboard screenshot
```

**Steps:**

1. Clone this repository: `git clone https://github.com/MrabetOussama0/Sentinel-Protocol.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the project root with your watsonx credentials (already in `.gitignore` — do **not** hardcode these values anywhere in source):
   ```
   WATSONX_API_KEY=your_ibm_cloud_api_key
   WATSONX_PROJECT_ID=your_watsonx_project_guid
   ```
   > **Region note:** the reasoning layer is configured for the `eu-de` (Frankfurt) endpoint. If your watsonx project is in a different region, update `WATSONX_URL` in [`sentinel/reasoning.py`](sentinel/reasoning.py) to match (e.g. `https://us-south.ml.cloud.ibm.com` for Dallas).
4. **Explore the core logic:** open `notebooks/sentinel_analysis.ipynb` in JupyterLab — it imports directly from the `sentinel` package to demonstrate the decision engine, safety gate, scenario simulator, and anomaly detection layer interactively.
5. **Run the live dashboard:** from the project root, run `streamlit run dashboard.py` and open the local URL it provides in your browser.

## Challenge Theme

Advance Space Exploration with AI — August Challenge

## How IBM Bob Was Used

IBM Bob was used across the full build of Sentinel Protocol's backend: designing and implementing the `classify_threat()` decision engine and its `Threat` dataclass, building the tick-by-tick scenario simulator that feeds realistic escalating sensor data into the engine, and integrating IBM watsonx.ai's `granite-4-h-small` model to generate the natural-language reasoning layer. Bob proposed the LLM prompt template and API integration approach, surfaced configuration decisions (model selection, endpoint, project ID) for explicit approval rather than assuming defaults, and iterated on the watsonx integration after an initial deprecated-endpoint issue was identified and corrected. Bob was also used to refactor the project from a single exploratory notebook into a proper Python package (`sentinel/`), separating the decision engine, safety gate, reasoning layer, and simulator into their own modules, with the notebook reduced to a demonstration and validation layer that imports from the package — the same structure a production codebase would use. Finally, Bob built the anomaly detection layer: loading and cleaning the NASA SMAP/MSL dataset, engineering channel-level statistical features, training and evaluating the IsolationForest model against held-out real telemetry channels, and integrating its output with the existing decision engine and safety gate — including identifying and correcting a safety gap where unclassified anomalies were initially passed through as safe rather than defaulting to a conservative hold.