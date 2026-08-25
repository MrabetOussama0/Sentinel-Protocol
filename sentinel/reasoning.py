"""
sentinel.reasoning
==================
IBM watsonx.ai integration: mission-log generation (generate_reasoning)
and Earth-facing block reports (_make_block_report).

Credentials are loaded from a .env file:
    WATSONX_API_KEY=<IBM Cloud API key>
    WATSONX_PROJECT_ID=<watsonx.ai project GUID>
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

WATSONX_URL      = "https://eu-de.ml.cloud.ibm.com"
WATSONX_MODEL_ID = "ibm/granite-4-h-small"

_CHAT_PARAMS = {
    "max_tokens":         80,
    "temperature":        0.3,
    "repetition_penalty": 1.05,
}

_BLOCK_CHAT_PARAMS = {
    "max_tokens":         90,
    "temperature":        0.3,
    "repetition_penalty": 1.05,
}

_wx_model      = None
_WATSONX_READY = False


def _init() -> bool:
    """Lazily initialise the watsonx ModelInference client. Returns True if ready."""
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
            model_id    = WATSONX_MODEL_ID,
            credentials = Credentials(url=WATSONX_URL, api_key=api_key),
            project_id  = project_id,
        )
        _WATSONX_READY = True
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mission-log generation
# ---------------------------------------------------------------------------

_REASONING_SYSTEM = (
    "You are the autonomous reasoning system of a planetary rover named Sentinel. "
    "Write a single professional mission-log sentence (maximum 40 words) explaining "
    "the decision made. Be factual, precise, and terse \u2014 like a flight engineer "
    "writing a flight log entry. Output only the log sentence, nothing else."
)

_REASONING_USER = (
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


def generate_reasoning(tick_data: dict) -> str:
    """Generate a mission-log sentence for one tick via watsonx.ai.

    Parameters
    ----------
    tick_data : dict with keys threat_type, sensors, time_to_harm_s,
                round_trip_s, ratio, tier, action.

    Returns
    -------
    str — mission-log sentence, or a fallback message if watsonx unavailable.
    """
    if not _init():
        return "(watsonx not configured \u2014 check .env credentials)"
    try:
        resp = _wx_model.chat(
            messages=[
                {"role": "system", "content": _REASONING_SYSTEM},
                {"role": "user",   "content": _REASONING_USER.format(**tick_data)},
            ],
            params=_CHAT_PARAMS,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(model error: {e})"


# ---------------------------------------------------------------------------
# Block-report generation (used by safety_gate via validate_command)
# ---------------------------------------------------------------------------

_BLOCK_SYSTEM = (
    "You are the autonomous safety system of a planetary rover named Sentinel. "
    "A command from Earth has been blocked because it conflicts with an active hazard. "
    "Write a single professional sentence (maximum 45 words) reporting the block back to Earth. "
    "Be factual and terse \u2014 like a flight engineer writing a status update. "
    "Output only the sentence, nothing else."
)

_BLOCK_USER = (
    "Blocked command : {command}\n"
    "Active threat   : {threat_type}\n"
    "Sensor state    : {sensors}\n"
    "Conflict reason : {reason}\n\n"
    "Write the status report back to Earth."
)


def make_block_report(command: str, threat_type: str, sensors: dict, reason: str) -> str:
    """Generate a plain-language Earth-facing block report via watsonx.ai."""
    if not _init():
        return "(watsonx not configured \u2014 check .env credentials)"
    try:
        resp = _wx_model.chat(
            messages=[
                {"role": "system", "content": _BLOCK_SYSTEM},
                {"role": "user",   "content": _BLOCK_USER.format(
                    command=command, threat_type=threat_type,
                    sensors=str(sensors), reason=reason)},
            ],
            params=_BLOCK_CHAT_PARAMS,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(model error: {e})"
