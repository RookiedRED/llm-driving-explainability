from __future__ import annotations
from typing import Dict, List, Tuple, Any

ALLOWED_ACTIONS = {"brake", "slow_down", "keep", "lane_change_left", "lane_change_right"}

def check_prediction(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    record: {"scene","timestamp_us","state_risk","model_output"}
    """
    issues: List[str] = []
    out = record.get("model_output", {})
    state_risk = record.get("state_risk", {})

    # 1) action allowed
    action = out.get("action")
    if action not in ALLOWED_ACTIONS:
        issues.append(f"Invalid action: {action}")

    # 2) evidence keys should exist and match state values (if present)
    ev = out.get("evidence", {})
    for k, v in ev.items():
        # if model provided a value, we can compare (when state has same key)
        if k in state_risk:
            sv = state_risk.get(k)
            # allow None in both
            if v is None and sv is None:
                continue
            # if one is None and other isn't
            if (v is None) != (sv is None):
                issues.append(f"Evidence mismatch for {k}: pred={v} vs state={sv}")
                continue
            # numeric compare with tolerance
            if isinstance(v, (int, float)) and isinstance(sv, (int, float)):
                if abs(v - sv) > 0.2:  # tolerance
                    issues.append(f"Evidence numeric mismatch for {k}: pred={v} vs state={sv}")
            else:
                if v != sv:
                    issues.append(f"Evidence mismatch for {k}: pred={v} vs state={sv}")

    ok = len(issues) == 0
    return ok, issues