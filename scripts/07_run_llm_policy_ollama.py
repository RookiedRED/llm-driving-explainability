import json
import re
from pathlib import Path
from time import perf_counter
import requests

from src.reasoning.prompt import build_policy_prompt
from src.reasoning.guardrails import apply_guardrails

IN_PATH = Path("data/derived/driving_states_v2.jsonl")
OUT_PATH = Path("data/derived/predictions_policy_ollama_v1.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

MAX_RETRIES = 3
TIMEOUT_S = 120


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found.")
        return json.loads(m.group(0))


def call_ollama(prompt: str) -> tuple[dict, float]:
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    t0 = perf_counter()
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_S)
    dt = perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = ok_n = override_n = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            state = json.loads(line)

            # minimal state exposed to LLM (structured)
            state_for_llm = {
                "scene": state["scene"],
                "timestamp_us": state["timestamp_us"],
                "ego": state.get("ego", {}),
                "objects": state.get("objects", []),
                "risk": state.get("risk", {}),
                "risk_physics": state.get("risk_physics", {}),
            }

            state_risk = {
                "risk_level_ttc": state.get("risk", {}).get("level"),
                "min_ttc_s": state.get("risk", {}).get("min_ttc_s"),
                "risk_level_physics": state.get("risk_physics", {}).get("level"),
                "closest_front_object_m": state.get("risk_physics", {}).get("closest_front_object_m"),
                "required_deceleration_mps2": state.get("risk_physics", {}).get("required_deceleration_mps2"),
            }

            prompt = build_policy_prompt(state_for_llm)

            last_err = None
            latency_s = None
            parsed = None
            raw = None

            for _ in range(MAX_RETRIES):
                try:
                    resp, latency_s = call_ollama(prompt)
                    raw = resp.get("response", "")
                    parsed = extract_json(raw)
                    break
                except Exception as e:
                    last_err = str(e)

            if parsed is None:
                record = {
                    "scene": state["scene"],
                    "timestamp_us": state["timestamp_us"],
                    "state_risk": state_risk,
                    "model": {"provider": "ollama", "name": MODEL},
                    "latency_ms": None if latency_s is None else round(latency_s * 1000, 2),
                    "policy": None,
                    "final_action": "slow_down",
                    "override_applied": True,
                    "override_reason": "Model failure; fallback slow_down",
                    "error": last_err,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                continue

            proposed = parsed.get("proposed_action")
            final_action, override, reason = apply_guardrails(state_risk, proposed)

            if override:
                override_n += 1

            record = {
                "scene": state["scene"],
                "timestamp_us": state["timestamp_us"],
                "state_risk": state_risk,
                "model": {"provider": "ollama", "name": MODEL},
                "latency_ms": None if latency_s is None else round(latency_s * 1000, 2),
                "policy": {
                    "proposed_action": proposed,
                    "rationale": parsed.get("rationale", []),
                    "confidence": parsed.get("confidence", None),
                },
                "final_action": final_action,
                "override_applied": override,
                "override_reason": reason if override else None,
                "raw_response_preview": None if raw is None else raw[:300],
                "error": None,
            }

            ok_n += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"✅ Wrote {n} records to {OUT_PATH}")
    print(f"✅ Parsed JSON success: {ok_n}/{n} ({(ok_n/n)*100:.2f}%)")
    print(f"🛡️ Guardrail overrides: {override_n}/{n} ({(override_n/n)*100:.2f}%)")


if __name__ == "__main__":
    main()