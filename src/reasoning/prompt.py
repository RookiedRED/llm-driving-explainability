import json

ALLOWED_ACTIONS = [
    "brake",
    "slow_down",
    "keep",
    "lane_change_left",
    "lane_change_right"
]

def build_policy_prompt(state: dict) -> str:
    schema = {
        "proposed_action": f"One of {ALLOWED_ACTIONS}",
        "rationale": [
            "Short bullets grounded in the input state",
            "No invented numbers or objects"
        ],
        "confidence": "float 0..1"
    }

    rules = f"""
        Rules (STRICT):
        - Decide a proposed_action based ONLY on the given state.
        - Do NOT invent numbers, objects, or signals not present.
        - Do NOT output evidence fields; the system will attach evidence and run safety guardrails.
        - Output MUST be valid JSON only (no markdown, no extra text).
        - Allowed actions: {ALLOWED_ACTIONS}

        Decision objective:
        - Maintain safety (highest priority).
        - Avoid unnecessary braking in low-risk situations.
        - Prefer smooth driving (slow_down over brake when sufficient).
        - Only use brake when strong deceleration is required.

        """.strip()
    
    decision_objective = """
        Decision objective:
        - Maintain safety (highest priority).
        - Avoid unnecessary braking in low-risk situations.
        - Prefer smooth driving (slow_down over brake when sufficient).
        - Only use brake when strong deceleration is required.
        """.strip() 
    
    return f"""{rules}

        {decision_objective}

        Driving state:
        {json.dumps(state, ensure_ascii=False)}

        Return JSON with this schema:
        {json.dumps(schema, ensure_ascii=False)}
        """.strip()