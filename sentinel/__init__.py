"""
sentinel
========
Sentinel Protocol — Autonomous Decision-Time-Budget Engine for planetary rovers.

Submodules
----------
sentinel.decision_engine  — DecisionTier, Threat, classify_threat(), THREAT_CONSERVATISM
sentinel.simulator        — TickState, run_scenario(), choose_holding_action()
sentinel.safety_gate      — is_action_safe(), validate_command(), blackout_survival_loop()
sentinel.reasoning        — generate_reasoning(), make_block_report()
"""

from sentinel.decision_engine import (
    DecisionTier,
    Threat,
    ThreatType,
    THREAT_CONSERVATISM,
    classify_threat,
)
from sentinel.simulator import (
    TickState,
    run_scenario,
    choose_holding_action,
)
from sentinel.safety_gate import (
    SafetyCheckResult,
    ValidationResult,
    SurvivalStep,
    is_action_safe,
    validate_command,
    blackout_survival_loop,
)
from sentinel.reasoning import generate_reasoning, make_block_report

__all__ = [
    "DecisionTier", "Threat", "ThreatType", "THREAT_CONSERVATISM", "classify_threat",
    "TickState", "run_scenario", "choose_holding_action",
    "SafetyCheckResult", "ValidationResult", "SurvivalStep",
    "is_action_safe", "validate_command", "blackout_survival_loop",
    "generate_reasoning", "make_block_report",
]
