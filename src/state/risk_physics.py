from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class PhysicsRiskConfig:
    reaction_time_s: float = 0.6
    comfort_decel_mps2: float = 3.0
    hard_decel_mps2: float = 6.0
    emergency_decel_mps2: float = 8.0
    min_distance_m: float = 0.1  # avoid division by zero


def braking_distance(v_mps: float, decel_mps2: float) -> float:
    # d = v^2 / (2a)
    return (v_mps * v_mps) / (2.0 * max(decel_mps2, 1e-6))


def compute_physics_risk(
    ego_speed_mps: Optional[float],
    closest_front_dist_m: Optional[float],
    cfg: PhysicsRiskConfig = PhysicsRiskConfig(),
) -> Dict:
    """
    Physics-first risk estimation.
    Returns a dict suitable to embed into your DrivingState JSON.
    """
    if ego_speed_mps is None or closest_front_dist_m is None:
        return {
            "level": "unknown",
            "reason": "Missing ego_speed_mps or closest_front_dist_m",
        }

    v = max(0.0, ego_speed_mps)
    d = max(cfg.min_distance_m, closest_front_dist_m)

    reaction_dist = v * cfg.reaction_time_s
    brake_dist_comfort = braking_distance(v, cfg.comfort_decel_mps2)
    brake_dist_hard = braking_distance(v, cfg.hard_decel_mps2)

    stopping_dist_comfort = reaction_dist + brake_dist_comfort
    stopping_dist_hard = reaction_dist + brake_dist_hard

    # Required decel to stop within distance d after reaction time
    # Remaining distance after reaction:
    remaining = max(cfg.min_distance_m, d - reaction_dist)
    required_decel = (v * v) / (2.0 * remaining)

    # Collision margin under hard braking (more relevant for safety)
    margin_hard = d - stopping_dist_hard

    # Risk logic (Balanced)
    # - If even HARD braking can't avoid collision (margin_hard < 0): HIGH
    # - Else if requires > HARD decel: HIGH
    # - Else if requires > COMFORT decel: MEDIUM
    # - Else LOW
    if margin_hard < 0:
        level = "high"
        reason = "Stopping distance (hard) exceeds available distance"
    elif required_decel > cfg.hard_decel_mps2:
        level = "high"
        reason = "Required deceleration exceeds hard braking threshold"
    elif required_decel > cfg.comfort_decel_mps2:
        level = "medium"
        reason = "Requires stronger-than-comfort braking"
    else:
        level = "low"
        reason = "Comfort braking sufficient"

    # Also flag extreme cases
    emergency_flag = required_decel > cfg.emergency_decel_mps2

    return {
        "closest_front_object_m": round(d, 2),
        "ego_speed_mps": round(v, 2),
        "reaction_time_s": cfg.reaction_time_s,
        "reaction_distance_m": round(reaction_dist, 2),
        "braking_distance_comfort_m": round(brake_dist_comfort, 2),
        "braking_distance_hard_m": round(brake_dist_hard, 2),
        "stopping_distance_comfort_m": round(stopping_dist_comfort, 2),
        "stopping_distance_hard_m": round(stopping_dist_hard, 2),
        "collision_margin_hard_m": round(margin_hard, 2),
        "required_deceleration_mps2": round(required_decel, 2),
        "emergency_decel_flag": bool(emergency_flag),
        "level": level,
        "reason": reason,
    }