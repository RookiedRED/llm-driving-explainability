import json
from pathlib import Path
from src.eval.groundedness import check_prediction

IN_PATH = Path("data/derived/predictions_stub_v1.jsonl")

def main():
    n = 0
    ok_n = 0
    bad_examples = []

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ok, issues = check_prediction(record)
            n += 1
            if ok:
                ok_n += 1
            elif len(bad_examples) < 5:
                bad_examples.append((record["scene"], record["timestamp_us"], issues))

    print(f"Groundedness pass rate: {ok_n}/{n} ({(ok_n/n)*100:.2f}%)")
    if bad_examples:
        print("\nSample failures:")
        for scene, ts, issues in bad_examples:
            print(f"- {scene} @ {ts}: {issues}")

if __name__ == "__main__":
    main()