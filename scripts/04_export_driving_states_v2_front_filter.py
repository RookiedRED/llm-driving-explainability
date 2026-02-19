import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from src.state.risk_physics import compute_physics_risk

NUSCENES_ROOT = "data/nuscenes"
VERSION = "v1.0-mini"

OUT_DIR = Path("data/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "driving_states_v2.jsonl"


def dist_xy(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def simplify_category(category_name: str) -> str:
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle."):
        return "vehicle"
    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if category_name.startswith("movable_object.barrier"):
        return "barrier"
    return "other"

def yaw_from_quat(q: List[float]) -> float:
    # nuScenes stores quaternion as [w, x, y, z]
    quat = Quaternion(q)
    # yaw from rotation matrix
    R = quat.rotation_matrix
    yaw = math.atan2(R[1, 0], R[0, 0])
    return yaw  # radians

def bearing_deg(ego_xy: List[float], ego_yaw: float, obj_xy: List[float]) -> float:
    # angle between ego forward direction and vector to object
    vx = obj_xy[0] - ego_xy[0]
    vy = obj_xy[1] - ego_xy[1]
    # forward unit vector
    fx = math.cos(ego_yaw)
    fy = math.sin(ego_yaw)
    # dot & cross to get signed angle
    dot = fx * vx + fy * vy
    cross = fx * vy - fy * vx
    ang = math.atan2(cross, dot)  # [-pi, pi]
    return math.degrees(ang)

def risk_level_from_ttc(min_ttc: Optional[float]) -> Tuple[str, str]:
    if min_ttc is None:
        return ("unknown", "No valid TTC computed")
    if min_ttc < 1.5:
        return ("high", "TTC < 1.5s")
    if min_ttc < 3.0:
        return ("medium", "TTC < 3.0s")
    return ("low", "TTC >= 3.0s")


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

                cam_token = sample["data"]["CAM_FRONT"]
                cam_sd = nusc.get("sample_data", cam_token)
                ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

                ego_xy = ego_pose["translation"][:2]
                ego_yaw = yaw_from_quat(ego_pose["rotation"])
                ts_us = sample["timestamp"]

                # ego speed from ego pose delta
                ego_speed_mps = None
                dt_s = None
                if prev_ego_xy is not None and prev_ts_us is not None:
                    dt_s = (ts_us - prev_ts_us) / 1_000_000.0
                    if dt_s > 0:
                        ego_speed_mps = dist_xy(ego_xy, prev_ego_xy) / dt_s

                objects = []
                min_ttc = None

                # front filter: keep objects within +/- 35 degrees
                FRONT_DEG = 35.0
                MAX_DIST = 60.0

                for ann_token in sample["anns"]:
                    ann = nusc.get("sample_annotation", ann_token)
                    cat = simplify_category(ann["category_name"])
                    obj_xy = ann["translation"][:2]

                    d = dist_xy(ego_xy, obj_xy)
                    if d > MAX_DIST:
                        continue

                    ang = bearing_deg(ego_xy, ego_yaw, obj_xy)
                    in_front = abs(ang) <= FRONT_DEG

                    # relative speed approx (distance change), only if we have prev
                    rel_speed_mps = None
                    ttc = None
                    if prev_ego_xy is not None and prev_ts_us is not None and dt_s and dt_s > 0:
                        prev_d = dist_xy(prev_ego_xy, obj_xy)
                        rel_speed_mps = (d - prev_d) / dt_s  # negative => closing

                        if in_front and rel_speed_mps < -0.1:
                            ttc = d / (-rel_speed_mps)
                            if min_ttc is None or ttc < min_ttc:
                                min_ttc = ttc

                    objects.append({
                        "type": cat,
                        "distance_m": round(d, 2),
                        "bearing_deg": round(ang, 1),
                        "in_front": in_front,
                        "rel_speed_mps": None if rel_speed_mps is None else round(rel_speed_mps, 2),
                        "ttc_s": None if ttc is None else round(ttc, 2),
                    })

                # ✅ Find closest object in front cone
                closest_front = None
                for o in objects:
                    if o["in_front"]:
                        d = o["distance_m"]
                        if closest_front is None or d < closest_front:
                            closest_front = d

                # ✅ Physics-first risk estimation
                risk_physics = compute_physics_risk(ego_speed_mps, closest_front)

                level, reason = risk_level_from_ttc(min_ttc)

                # keep nearest 30 for readability
                objects_sorted = sorted(objects, key=lambda o: o["distance_m"])[:30]

                state = {
                    "dataset": "nuscenes",
                    "version": VERSION,
                    "scene": scene_name,
                    "timestamp_us": ts_us,
                    "ego": {
                        "speed_mps": None if ego_speed_mps is None else round(ego_speed_mps, 2),
                        "yaw_deg": round(math.degrees(ego_yaw), 1),
                    },
                    "objects": objects_sorted,
                    "risk": {
                        "min_ttc_s": None if min_ttc is None else round(min_ttc, 2),
                        "level": level,
                        "reason": reason,
                        "front_cone_deg": FRONT_DEG,
                    },
                    "risk_physics": risk_physics
                }

                f.write(json.dumps(state, ensure_ascii=False) + "\n")
                total_written += 1

                prev_ego_xy = ego_xy
                prev_ts_us = ts_us
                sample_token = sample["next"]

    print(f"✅ Wrote {total_written} states to {OUT_PATH}")


if __name__ == "__main__":
    main()