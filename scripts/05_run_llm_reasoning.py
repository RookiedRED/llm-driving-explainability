import json
from pathlib import Path
from time import perf_counter

IN_PATH = Path("data/derived/driving_states_v2.jsonl")
OUT_PATH = Path("data/derived/llm_outputs_v1.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def stub_llm(state: dict) -> dict:
    # Simple baseline: physics-first decision
    rp = state.get("risk_physics", {})
    level = rp.get("level", "unknown")

    if level == "high":
        action = "brake"
    elif level == "medium":
        action = "slow_down"
    else:
        action = "keep"

    return {
        "action": action,
        "explanation": [
            f"Physics risk level is {level}.",
            "Decision is based on stopping distance vs. closest front object distance."
        ],
        "evidence": {
            "risk_level_physics": rp.get("level"),
            "closest_front_object_m": rp.get("closest_front_object_m"),
            "required_deceleration_mps2": rp.get("required_deceleration_mps2"),
        },
        "safety_notes": ["This is a baseline stub without LLM reasoning."],
        "confidence": 0.6
    }

def main():
    t0 = perf_counter()
    n = 0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            state = json.loads(line)
            out = stub_llm(state)
            record =  {
                "scene": state["scene"],
                "timestamp_us": state["timestamp_us"],
                "state_risk": {
                    "ttc_level": state.get("risk", {}).get("level"),
                    "min_ttc_s": state.get("risk", {}).get("min_ttc_s"),
                    "physics_level": state.get("risk_physics", {}).get("level"),
                    "closest_front_object_m": state.get("risk_physics", {}).get("closest_front_object_m"),
                    "required_deceleration_mps2": state.get("risk_physics", {}).get("required_deceleration_mps2"),
                },
                "model_output": out
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    dt = perf_counter() - t0
    print(f"✅ Wrote {n} outputs to {OUT_PATH}")
    print(f"⏱️ Total time: {dt:.3f}s | avg {(dt/n)*1000:.2f} ms/sample")

if __name__ == "__main__":
    main()