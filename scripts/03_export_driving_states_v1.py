import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from nuscenes.nuscenes import NuScenes


NUSCENES_ROOT = "data/nuscenes"
VERSION = "v1.0-mini"

OUT_DIR = Path("data/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "driving_states_v1.jsonl"


# --- Helpers ---

def dist_xy(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def simplify_category(category_name: str) -> str:
    # nuScenes categories look like: vehicle.car, human.pedestrian.adult, movable_object.trafficcone
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("human.") and "pedestrian" not in category_name:
        return "human"
    if category_name.startswith("vehicle."):
        return "vehicle"
    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if category_name.startswith("movable_object.barrier"):
        return "barrier"
    if category_name.startswith("animal"):
        return "animal"
    return "other"

def risk_level_from_ttc(min_ttc: Optional[float]) -> Tuple[str, str]:
    if min_ttc is None:
        return ("unknown", "No valid TTC computed")
    if min_ttc < 1.5:
        return ("high", "TTC < 1.5s")
    if min_ttc < 3.0:
        return ("medium", "TTC < 3.0s")
    return ("low", "TTC >= 3.0s")


# --- Main Export ---

def main():
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)

    total_written = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for scene in nusc.scene:
            scene_name = scene["name"]
            sample_token = scene["first_sample_token"]

            prev_ego_xy = None
            prev_ts_us = None

            while sample_token:
                sample = nusc.get("sample", sample_token)

                # Use CAM_FRONT sample_data for ego pose (stable choice)
                cam_token = sample["data"]["CAM_FRONT"]
                cam_sd = nusc.get("sample_data", cam_token)
                ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
                ego_xy = ego_pose["translation"][:2]

                ts_us = sample["timestamp"]

                # Ego speed: approximate from consecutive ego positions
                ego_speed_mps = None
                if prev_ego_xy is not None and prev_ts_us is not None:
                    dt_s = (ts_us - prev_ts_us) / 1_000_000.0
                    if dt_s > 0:
                        ego_speed_mps = dist_xy(ego_xy, prev_ego_xy) / dt_s

                # Objects + relative speed (approx): distance change per dt
                objects = []
                min_ttc = None

                for ann_token in sample["anns"]:
                    ann = nusc.get("sample_annotation", ann_token)
                    cat = simplify_category(ann["category_name"])
                    obj_xy = ann["translation"][:2]
                    d = dist_xy(ego_xy, obj_xy)

                    # Filter: keep nearby objects (tune later)
                    if d > 60:
                        continue

                    # relative speed approx (negative means closing in) using distance delta
                    rel_speed_mps = None
                    if prev_ego_xy is not None and prev_ts_us is not None:
                        dt_s = (ts_us - prev_ts_us) / 1_000_000.0
                        if dt_s > 0:
                            # Approx: assume object stationary in world in this quick version
                            prev_d = dist_xy(prev_ego_xy, obj_xy)
                            rel_speed_mps = (d - prev_d) / dt_s  # negative => closing

                    # TTC approx if closing
                    ttc = None
                    if rel_speed_mps is not None and rel_speed_mps < -0.1:
                        ttc = d / (-rel_speed_mps)
                        if min_ttc is None or ttc < min_ttc:
                            min_ttc = ttc

                    objects.append({
                        "type": cat,
                        "distance_m": round(d, 2),
                        "rel_speed_mps": None if rel_speed_mps is None else round(rel_speed_mps, 2),
                        "ttc_s": None if ttc is None else round(ttc, 2),
                    })

                level, reason = risk_level_from_ttc(min_ttc)

                state = {
                    "dataset": "nuscenes",
                    "version": VERSION,
                    "scene": scene_name,
                    "timestamp_us": ts_us,
                    "ego": {
                        "speed_mps": None if ego_speed_mps is None else round(ego_speed_mps, 2)
                    },
                    "objects": sorted(objects, key=lambda o: o["distance_m"])[:30],
                    "risk": {
                        "min_ttc_s": None if min_ttc is None else round(min_ttc, 2),
                        "level": level,
                        "reason": reason
                    }
                }

                f.write(json.dumps(state, ensure_ascii=False) + "\n")
                total_written += 1

                prev_ego_xy = ego_xy
                prev_ts_us = ts_us
                sample_token = sample["next"]

    print(f"✅ Wrote {total_written} states to {OUT_PATH}")

if __name__ == "__main__":
    main()