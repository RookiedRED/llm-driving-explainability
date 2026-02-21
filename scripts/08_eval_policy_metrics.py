import json
from pathlib import Path
from collections import Counter
import math

IN_PATH = Path("data/derived/predictions_policy_ollama_v1.jsonl")

def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def main():
    n = 0
    proposed_ctr = Counter()
    final_ctr = Counter()
    physics_ctr = Counter()
    override_ctr = Counter()

    latencies = []

    # Cross table: physics_level -> proposed_action counts
    cross = {}

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1

            phys = r.get("state_risk", {}).get("risk_level_physics", "unknown")
            physics_ctr[phys] += 1

            pol = r.get("policy") or {}
            proposed = pol.get("proposed_action", "none")
            final = r.get("final_action", "none")

            proposed_ctr[proposed] += 1
            final_ctr[final] += 1

            override = bool(r.get("override_applied", False))
            override_ctr["override" if override else "no_override"] += 1

            lat = r.get("latency_ms")
            if isinstance(lat, (int, float)):
                latencies.append(lat)

            cross.setdefault(phys, Counter())
            cross[phys][proposed] += 1

    print(f"\nTotal records: {n}")

    print("\nPhysics risk distribution:")
    for k, v in physics_ctr.most_common():
        print(f"  {k:8s}: {v} ({v/n*100:.2f}%)")

    print("\nProposed action distribution:")
    for k, v in proposed_ctr.most_common():
        print(f"  {k:18s}: {v} ({v/n*100:.2f}%)")

    print("\nFinal action distribution:")
    for k, v in final_ctr.most_common():
        print(f"  {k:18s}: {v} ({v/n*100:.2f}%)")

    print("\nOverride rate:")
    ov = override_ctr["override"]
    print(f"  overrides: {ov}/{n} ({ov/n*100:.2f}%)")

    if latencies:
        print("\nLatency (ms):")
        print(f"  mean: {sum(latencies)/len(latencies):.2f}")
        print(f"  p50 : {percentile(latencies, 50):.2f}")
        print(f"  p90 : {percentile(latencies, 90):.2f}")
        print(f"  p99 : {percentile(latencies, 99):.2f}")

    print("\nPhysics level -> Proposed action (counts):")
    for phys, ctr in cross.items():
        top = ", ".join([f"{a}:{c}" for a, c in ctr.most_common(5)])
        print(f"  {phys:8s}: {top}")

if __name__ == "__main__":
    main()