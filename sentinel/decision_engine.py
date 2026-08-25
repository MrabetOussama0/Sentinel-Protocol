"""
sentinel.decision_engine
========================
Core threat classification: Threat dataclass, DecisionTier enum,
THREAT_CONSERVATISM table, and classify_threat().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class DecisionTier(Enum):
    GREEN  = "GREEN"    # safe to wait for Earth
    YELLOW = "YELLOW"   # hold + notify Earth
    RED    = "RED"      # act now, notify later


ThreatType = Literal[
    "cliff_edge",
    "dust_storm",
    "battery_critical",
    "rockfall",
    "comms_blackout",
]


@dataclass
class Threat:
    """Represents a detected hazard encountered by the rover.

    Attributes
    ----------
    threat_type    : One of the five recognised threat categories.
    time_to_harm_s : Estimated seconds until irreversible damage if no action taken.
    comm_delay_s   : One-way communication delay to Earth in seconds.
    """
    threat_type:    str
    time_to_harm_s: float
    comm_delay_s:   float

    @property
    def round_trip_s(self) -> float:
        """Full round-trip comm delay (signal to Earth + command back)."""
        return self.comm_delay_s * 2

    @property
    def time_margin_ratio(self) -> float:
        """Ratio of time-to-harm to round-trip delay. >2 → GREEN, 1-2 → YELLOW, ≤1 → RED."""
        if self.round_trip_s == 0:
            return float("inf")
        return self.time_to_harm_s / self.round_trip_s


# ---------------------------------------------------------------------------
# Threat-specific conservatism multipliers
# ---------------------------------------------------------------------------

THREAT_CONSERVATISM: dict[str, float] = {
    "cliff_edge":           0.80,   # sensor noise → be conservative
    "dust_storm":           0.90,   # storm intensity can escalate quickly
    "dust_storm_slow":      0.90,   # same physics, slower ramp
    "battery_critical":     0.95,   # discharge rate is fairly predictable
    "rockfall":             0.70,   # highly dynamic, worst-case bias
    "comms_blackout":       1.00,   # predictable orbital geometry
    "full_blackout":        0.90,   # relay lost; battery drain is predictable
    # Unknown anomaly detected by IsolationForest but not matched to a known
    # threat type.  Worst-case bias: treat as potentially as fast-escalating as
    # rockfall.  All risky actions are blocked by safety_gate until Earth
    # confirms the nature of the anomaly.
    "unclassified_anomaly": 0.75,
}


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_threat(
    threat_type:    str,
    time_to_harm_s: float,
    comm_delay_s:   float,
) -> DecisionTier:
    """Classify a rover threat into a decision tier.

    Parameters
    ----------
    threat_type    : Category of the detected hazard.
    time_to_harm_s : Estimated seconds to irreversible harm.
    comm_delay_s   : One-way comm latency to Earth in seconds.

    Returns
    -------
    DecisionTier
        GREEN  — time_to_harm > 2 × RTT  (adjusted for conservatism)
        YELLOW — 1 × RTT < time_to_harm ≤ 2 × RTT
        RED    — time_to_harm ≤ RTT
    """
    if threat_type not in THREAT_CONSERVATISM:
        raise ValueError(f"Unknown threat type: {threat_type!r}")

    conservatism = THREAT_CONSERVATISM[threat_type]
    threat = Threat(
        threat_type=threat_type,
        time_to_harm_s=time_to_harm_s * conservatism,
        comm_delay_s=comm_delay_s,
    )
    ratio = threat.time_margin_ratio

    if ratio > 2.0:
        return DecisionTier.GREEN
    elif ratio > 1.0:
        return DecisionTier.YELLOW
    else:
        return DecisionTier.RED
