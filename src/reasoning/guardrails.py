from __future__ import annotations
from typing import Dict, Any, Tuple

ALLOWED_ACTIONS = {"brake", "slow_down", "keep", "lane_change_left", "lane_change_right"}

def apply_guardrails(state_risk: Dict[str, Any], proposed_action: str) -> Tuple[str, bool, str]:
    """
    Returns: (final_action, override_applied, override_reason)
    """
    if proposed_action not in ALLOWED_ACTIONS:
        return "slow_down", True, "Invalid proposed_action"

    physics_level = state_risk.get("risk_level_physics", "unknown")

    # Rule: high => must brake
    if physics_level == "high":
        if proposed_action != "brake":
            return "brake", True, "Physics risk is high; braking required"
        return "brake", False, ""

    # Rule: medium => at least slow_down (no keep)
    if physics_level == "medium":
        if proposed_action in {"keep", "lane_change_left", "lane_change_right"}:
            return "slow_down", True, "Physics risk is medium; cannot keep speed"
        return proposed_action, False, ""

    # Low/unknown: allow keep/slow_down/brake, but block lane changes for v1
    if proposed_action in {"lane_change_left", "lane_change_right"}:
        # Conservative v1: disallow lane changes without gap checking
        fallback = "slow_down" if physics_level in {"unknown"} else "keep"
        return fallback, True, "Lane change disabled (no gap checking in v1)"

    return proposed_action, False, ""