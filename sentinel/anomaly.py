"""
sentinel.anomaly
================
IsolationForest-based anomaly detection trained on the NASA SMAP/MSL
Anomaly Detection Dataset (labeled_anomalies.csv).

Public API
----------
classify_sensor_pattern(sensor_state, window)
    Score a rover sensor reading dict.  Returns AnomalyResult.

train_and_save(data_path, model_path)
    Fit the model from scratch and persist it with joblib.

load_model(model_path)
    (Re-)load a persisted model.

evaluate(data_path, model_path, test_frac, random_state)
    Channel-level precision / recall / F1 against the SMAP/MSL labels.

Design notes
------------
The labeled_anomalies.csv file is metadata-only: it records which index
windows in each telemetry channel are anomalous, not the raw sensor values.
We therefore work with **channel-level pattern features** derived from the
anomaly-window geometry (fraction of anomalous points, window length
statistics, temporal clustering, anomaly class mix).

"Normal" proxy channels are synthesised by shuffling window geometry into
configurations statistically consistent with healthy telemetry (very low
anomaly fraction, short windows, early-series placement) so the model has
a negative class to contrast against.

At rover inference time, sensor_state readings are projected into the same
feature space using a rolling SensorWindow accumulator that tracks how
"anomalous" the last N readings look relative to a running baseline.  This
keeps the call signature simple — classify_sensor_pattern(sensor_state) —
while grounding predictions in the SMAP/MSL training distribution.

Integration with decision_engine / safety_gate
----------------------------------------------
If is_anomaly=True and the sensor_state does not match any known threat
type (cliff_edge, dust_storm, battery_critical, rockfall, comms_blackout),
the result carries threat_type="unclassified_anomaly" and tier=YELLOW.

"unclassified_anomaly" is registered in THREAT_CONSERVATISM with a
conservative multiplier of 0.75 (between rockfall and cliff_edge), and
safety_gate.is_action_safe() blocks all risky actions (movement, high-power,
antenna deployment, comms) whenever this threat is active.  Only _ALWAYS_SAFE
actions (hold, stop, emergency_full_stop) are permitted until Earth
confirms the nature of the anomaly.
"""

from __future__ import annotations

import ast
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from sentinel.decision_engine import DecisionTier, THREAT_CONSERVATISM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_PATH  = Path(__file__).parent.parent / "data" / "labeled_anomalies.csv"
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "data" / "anomaly_model.joblib"

# Known threat types from decision_engine — used to detect unclassified anomalies
_KNOWN_THREATS: frozenset[str] = frozenset(THREAT_CONSERVATISM.keys())

# Feature columns used for training and inference
_FEATURE_COLS = [
    "anom_frac",
    "n_windows",
    "mean_window_len",
    "max_window_len",
    "min_window_len",
    "std_window_len",
    "mean_start_norm",
    "min_start_norm",
    "mean_gap_norm",     # gap normalised by num_values
    "min_gap_norm",
    "frac_contextual",
]

# IsolationForest hyperparameters — tuned for the 82-channel dataset size
_IF_PARAMS = dict(
    n_estimators=200,
    contamination=0.15,   # ~12.5 % of channels are "deeply anomalous" in training set
    max_samples="auto",
    random_state=42,
)

# Window size for online feature accumulation at inference time
_WINDOW_SIZE = 30  # sensor ticks


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class AnomalyResult(NamedTuple):
    """Result returned by classify_sensor_pattern().

    Attributes
    ----------
    anomaly_score : float
        Raw IsolationForest decision-function score (higher = more normal,
        negative = anomalous).  Normalised to [0, 1] where 1 = most anomalous.
    is_anomaly    : bool
        True if the model predicts this reading as anomalous.
    threat_type   : str
        Matched known threat type, or "unclassified_anomaly" if flagged but
        not matching any known threat sensor signature.
    tier          : DecisionTier
        Decision tier to route through safety_gate.  YELLOW for unclassified.
    label         : str
        Human-readable label for the anomaly detection result.
    """
    anomaly_score : float
    is_anomaly    : bool
    threat_type   : str
    tier          : DecisionTier
    label         : str


# ---------------------------------------------------------------------------
# Feature engineering — channel level (training + evaluation)
# ---------------------------------------------------------------------------

def _parse_sequences(s: str) -> list[list[int]]:
    return ast.literal_eval(s)


def _parse_labels(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s)


def _channel_features(
    sequences: list[list[int]],
    labels:    list[str],
    n:         int,
) -> dict[str, float]:
    """Compute the 11 channel-level feature vector from anomaly window geometry."""
    lengths    = [e - s for s, e in sequences]
    gaps       = [(sequences[i][0] - sequences[i - 1][1]) for i in range(1, len(sequences))]
    start_norm = [s / n for s, _e in sequences]

    total_anom = sum(lengths)
    n_ctx      = labels.count("contextual")

    return {
        "anom_frac":       total_anom / n,
        "n_windows":       float(len(sequences)),
        "mean_window_len": float(np.mean(lengths)),
        "max_window_len":  float(np.max(lengths)),
        "min_window_len":  float(np.min(lengths)),
        "std_window_len":  float(np.std(lengths)) if len(lengths) > 1 else 0.0,
        "mean_start_norm": float(np.mean(start_norm)),
        "min_start_norm":  float(np.min(start_norm)),
        "mean_gap_norm":   float(np.mean(gaps) / n) if gaps else 1.0,
        "min_gap_norm":    float(np.min(gaps)  / n) if gaps else 1.0,
        "frac_contextual": float(n_ctx / len(sequences)),
    }


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the (X, y) feature matrix from labeled_anomalies.csv.

    All rows in the CSV are genuinely anomalous channels (y=1).
    We augment with synthetic "normal" proxy rows (y=0) so the model has
    a negative class.  Normals are generated by sampling n_values from the
    observed distribution and assigning minimal, early-placed anomaly windows
    that represent channels we would NOT flag (very low density).
    """
    rows, ys = [], []

    rng = np.random.default_rng(42)

    for _, row in df.iterrows():
        seqs   = _parse_sequences(row["anomaly_sequences"])
        labels = _parse_labels(row["class"])
        n      = int(row["num_values"])
        feats  = _channel_features(seqs, labels, n)
        rows.append(feats)
        ys.append(1)

        # Generate one synthetic normal per real channel:
        # - anom_frac < 0.01  (tiny window)
        # - single window, short (10–30 pts), placed in first 20% of series
        n_norm    = int(rng.choice([n, rng.integers(1000, 9000)]))
        win_len   = int(rng.integers(10, 31))
        win_start = int(rng.integers(0, max(1, int(n_norm * 0.20))))
        win_end   = win_start + win_len
        syn_seqs  = [[win_start, win_end]]
        syn_labs  = ["point"]
        rows.append(_channel_features(syn_seqs, syn_labs, n_norm))
        ys.append(0)

    X = pd.DataFrame(rows, columns=_FEATURE_COLS)
    y = pd.Series(ys, name="label")
    return X, y


# ---------------------------------------------------------------------------
# Model training & persistence
# ---------------------------------------------------------------------------

def train_and_save(
    data_path:  str | Path = _DEFAULT_DATA_PATH,
    model_path: str | Path = _DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Fit an IsolationForest pipeline on all channels and save with joblib.

    Parameters
    ----------
    data_path  : Path to labeled_anomalies.csv.
    model_path : Destination path for the serialised pipeline.

    Returns
    -------
    Fitted sklearn Pipeline (StandardScaler → IsolationForest).
    """
    df = pd.read_csv(data_path)
    X, _y = _build_feature_matrix(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(**_IF_PARAMS)),
    ])
    pipeline.fit(X)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


def load_model(model_path: str | Path = _DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a persisted pipeline from disk."""
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Lazy-loaded global model (trained once on first import call)
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    """Return the global pipeline, training it if not yet available."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    model_path = _DEFAULT_MODEL_PATH
    if Path(model_path).exists():
        _pipeline = load_model(model_path)
    else:
        _pipeline = train_and_save(_DEFAULT_DATA_PATH, model_path)
    return _pipeline


# ---------------------------------------------------------------------------
# Online sensor-window accumulator
# ---------------------------------------------------------------------------

@dataclass
class SensorWindow:
    """Rolling accumulator of rover sensor readings for inference-time feature extraction.

    Usage
    -----
    window = SensorWindow()
    for tick in simulation:
        window.push(tick.sensors)
        result = classify_sensor_pattern(tick.sensors, window)
    """
    _readings: list[dict] = field(default_factory=list)
    _size:     int        = _WINDOW_SIZE

    def push(self, sensor_state: dict) -> None:
        """Add the latest sensor reading; evict oldest if over window size."""
        self._readings.append(dict(sensor_state))
        if len(self._readings) > self._size:
            self._readings.pop(0)

    def __len__(self) -> int:
        return len(self._readings)


def _sensor_window_features(
    sensor_state: dict,
    window:       SensorWindow | None,
) -> dict[str, float]:
    """
    Project a rover sensor_state (and optional rolling window) into the
    11 channel-level feature space.

    Mapping
    -------
    The SMAP/MSL features describe anomaly-window geometry across a whole
    time series.  At inference, we approximate them from the rolling window:

    anom_frac        ← fraction of window ticks where any sensor deviates
                       > 2σ from the window mean (estimated anomalous rate)
    n_windows        ← number of contiguous "anomalous" runs in the window
    mean/max/min/std_window_len ← lengths of those runs
    mean/min_start_norm         ← position of anomalous runs in [0,1] timeline
    mean/min_gap_norm           ← normalised gaps between runs
    frac_contextual  ← fraction of anomalous ticks that are contextual
                       (i.e. isolated spikes = point; sustained = contextual)
    """
    # Require at least 3 readings in the window for meaningful z-score deviation;
    # with fewer readings all z-scores are zero by definition → safe baseline.
    if window is None or len(window) < 3:
        return {c: 0.0 for c in _FEATURE_COLS}

    readings = window._readings
    n        = len(readings)

    # Extract numeric values from each reading tick
    keys = [k for k in readings[-1] if isinstance(readings[-1][k], (int, float))]
    if not keys:
        # No numeric sensors — return safe baseline
        return {c: 0.0 for c in _FEATURE_COLS}

    # Build matrix: rows=ticks, cols=sensors
    matrix = np.array([[float(r.get(k, 0.0)) for k in keys] for r in readings])  # (n, d)

    # Per-sensor z-score deviation from window mean
    mean_  = matrix.mean(axis=0)
    std_   = matrix.std(axis=0) + 1e-9
    z      = np.abs((matrix - mean_) / std_)          # (n, d)
    any_z  = (z > 2.0).any(axis=1).astype(int)        # (n,) — tick anomalous if any sensor >2σ

    # Anomalous runs (contiguous ticks where any_z==1)
    anom_idx = np.where(any_z == 1)[0]
    if len(anom_idx) == 0:
        return {c: 0.0 for c in _FEATURE_COLS}

    # Identify contiguous runs
    runs = []
    run_start = anom_idx[0]
    prev      = anom_idx[0]
    for idx in anom_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((run_start, prev))
            run_start = idx
            prev      = idx
    runs.append((run_start, prev))

    lengths    = [e - s + 1 for s, e in runs]
    gaps       = [(runs[i][0] - runs[i - 1][1] - 1) for i in range(1, len(runs))]
    start_norm = [s / max(n - 1, 1) for s, _ in runs]

    anom_frac  = float(any_z.sum()) / n

    # Contextual: run length > 1; point: length == 1
    n_ctx = sum(1 for l in lengths if l > 1)

    return {
        "anom_frac":       anom_frac,
        "n_windows":       float(len(runs)),
        "mean_window_len": float(np.mean(lengths)),
        "max_window_len":  float(np.max(lengths)),
        "min_window_len":  float(np.min(lengths)),
        "std_window_len":  float(np.std(lengths)) if len(lengths) > 1 else 0.0,
        "mean_start_norm": float(np.mean(start_norm)),
        "min_start_norm":  float(np.min(start_norm)),
        "mean_gap_norm":   float(np.mean(gaps) / n) if gaps else 1.0,
        "min_gap_norm":    float(np.min(gaps)  / n) if gaps else 1.0,
        "frac_contextual": float(n_ctx / len(runs)),
    }


# ---------------------------------------------------------------------------
# Threat type matching from sensor state keys
# ---------------------------------------------------------------------------

_THREAT_SENSOR_KEYS: dict[str, frozenset[str]] = {
    "cliff_edge":       frozenset({"distance_m", "drift_speed_ms"}),
    "dust_storm":       frozenset({"wind_speed_ms", "dust_density_gcm3", "optical_depth"}),
    "battery_critical": frozenset({"charge_pct", "draw_pct_per_tick"}),
    "rockfall":         frozenset({"seismic_g", "debris_dist_m", "debris_speed_ms"}),
    "comms_blackout":   frozenset({"relay_elevation_deg", "effective_descent_rate"}),
}


def _match_threat_type(sensor_state: dict) -> str | None:
    """Return the best-matching known threat type from sensor keys, or None."""
    keys = frozenset(sensor_state.keys())
    best_threat, best_overlap = None, 0
    for threat, sig_keys in _THREAT_SENSOR_KEYS.items():
        overlap = len(keys & sig_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_threat  = threat
    # Require at least one matching key to commit to a known threat
    return best_threat if best_overlap >= 1 else None


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def classify_sensor_pattern(
    sensor_state: dict,
    window:       SensorWindow | None = None,
) -> AnomalyResult:
    """Score a rover sensor reading dict using the trained IsolationForest.

    Parameters
    ----------
    sensor_state : dict
        Current sensor readings from the rover (arbitrary numeric keys).
        Matches the format produced by sentinel.simulator physics models.
    window : SensorWindow | None
        Rolling window accumulator.  Must contain at least 3 readings
        for a valid score.  If None or len < 3, returns GREEN with
        score=0.0 (insufficient baseline — caller should accumulate more
        ticks before acting on the result).

    Returns
    -------
    AnomalyResult
        anomaly_score : float in [0, 1] — 1 = most anomalous
        is_anomaly    : bool
        threat_type   : known threat or "unclassified_anomaly"
        tier          : DecisionTier (YELLOW for unclassified)
        label         : human-readable description
    """
    # Without a window we have no deviation baseline — cannot make a meaningful
    # anomaly judgment, so default to GREEN (safe / insufficient data).
    if window is None or len(window) < 3:
        return AnomalyResult(
            anomaly_score = 0.0,
            is_anomaly    = False,
            threat_type   = "",
            tier          = DecisionTier.GREEN,
            label         = "normal (insufficient window — waiting for baseline)",
        )

    pipeline = _get_pipeline()

    feats = _sensor_window_features(sensor_state, window)
    X     = pd.DataFrame([feats], columns=_FEATURE_COLS)

    # decision_function: higher = more normal, negative = anomaly territory
    raw_score  = float(pipeline.decision_function(X)[0])
    prediction = int(pipeline.predict(X)[0])   # 1 = normal, -1 = anomaly (sklearn convention)

    is_anomaly = (prediction == -1)

    # Normalise raw_score to [0,1] where 1=most anomalous
    # Typical IF scores range roughly [-0.5, 0.5]; clamp before normalising
    normalised = float(np.clip(((-raw_score) + 0.5) / 1.0, 0.0, 1.0))

    if not is_anomaly:
        return AnomalyResult(
            anomaly_score = normalised,
            is_anomaly    = False,
            threat_type   = "",
            tier          = DecisionTier.GREEN,
            label         = "normal",
        )

    # Anomalous — try to match a known threat type from sensor keys
    matched_threat = _match_threat_type(sensor_state)

    if matched_threat is not None:
        # Route through decision_engine as normal (tier computed by caller)
        return AnomalyResult(
            anomaly_score = normalised,
            is_anomaly    = True,
            threat_type   = matched_threat,
            tier          = DecisionTier.YELLOW,   # caller may override via classify_threat
            label         = f"anomaly — matched threat: {matched_threat}",
        )
    else:
        # Unknown pattern — default to YELLOW, label "unclassified anomaly"
        return AnomalyResult(
            anomaly_score = normalised,
            is_anomaly    = True,
            threat_type   = "unclassified_anomaly",
            tier          = DecisionTier.YELLOW,
            label         = "unclassified anomaly — holding pending Earth contact",
        )


# ---------------------------------------------------------------------------
# Evaluation against SMAP/MSL labeled test set
# ---------------------------------------------------------------------------

def evaluate(
    data_path:    str | Path = _DEFAULT_DATA_PATH,
    model_path:   str | Path = _DEFAULT_MODEL_PATH,
    test_frac:    float      = 0.33,
    random_state: int        = 42,
) -> dict:
    """
    Evaluate channel-level anomaly detection accuracy against SMAP/MSL labels.

    Strategy
    --------
    * Build the full (X, y) matrix (real anomalous channels + synthetic normals).
    * Split 67/33 train/test — stratified to keep class balance.
    * Re-fit IsolationForest on training split only.
    * Predict on test split and compare to ground-truth labels.
    * Report accuracy, precision, recall, F1 and a per-class breakdown.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, support, report_str.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, classification_report,
    )

    df    = pd.read_csv(data_path)
    X, y  = _build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=random_state
    )

    # Fit fresh pipeline on train split (do NOT load cached model — evaluation must be honest)
    eval_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(**_IF_PARAMS)),
    ])
    eval_pipeline.fit(X_train)

    # IsolationForest predicts 1=normal, -1=anomaly; remap to 0/1
    raw_preds = eval_pipeline.predict(X_test)
    y_pred    = np.where(raw_preds == -1, 1, 0)
    y_true    = y_test.to_numpy()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    report_str = classification_report(
        y_true, y_pred,
        target_names=["normal (synthetic)", "anomalous (SMAP/MSL)"],
        zero_division=0,
    )

    return {
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "n_train":    len(y_train),
        "n_test":     len(y_test),
        "report_str": report_str,
    }
