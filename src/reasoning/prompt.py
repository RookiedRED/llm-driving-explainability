import json

ALLOWED_ACTIONS = [
    "brake",
    "slow_down",
    "keep",
    "lane_change_left",
    "lane_change_right"
]

def build_prompt(state: dict) -> str:
    # Keep prompt short and structured to reduce hallucination
    schema = {
        "action": f"One of {ALLOWED_ACTIONS}",
        "explanation": ["bullet strings, grounded in state only"],
        "evidence": {
            "risk_level_ttc": "state.risk.level",
            "min_ttc_s": "state.risk.min_ttc_s",
            "risk_level_physics": "state.risk_physics.level",
            "closest_front_object_m": "state.risk_physics.closest_front_object_m",
            "required_deceleration_mps2": "state.risk_physics.required_deceleration_mps2",
        },
        "safety_notes": ["short safety notes, no new facts"],
        "confidence": "float 0..1"
    }

    return f"""
You are an autonomous driving explainability system.

Rules:
- Use ONLY the provided driving state JSON. Do NOT invent objects, numbers, or signals not present.
- Output MUST be valid JSON only (no markdown, no extra text).
- Keep explanations short, concrete, and safety-focused.
- Choose an action from: {ALLOWED_ACTIONS}

Driving state:
{json.dumps(state, ensure_ascii=False)}

Return JSON with this schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()