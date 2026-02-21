import math
from nuscenes.nuscenes import NuScenes

NUSCENES_ROOT = "data/nuscenes"
VERSION = "v1.0-mini"

def dist_xy(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)

scene = nusc.scene[0]
sample_token = scene["first_sample_token"]
sample = nusc.get("sample", sample_token)

# Use front camera ego pose
cam_token = sample["data"]["CAM_FRONT"]
cam_sd = nusc.get("sample_data", cam_token)
ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
ego_xy = ego_pose["translation"][:2]

print("\nScene:", scene["name"])
print("Timestamp:", sample["timestamp"])
print("\nNearby objects:")

for ann_token in sample["anns"]:
    ann = nusc.get("sample_annotation", ann_token)
    category = ann["category_name"]
    obj_xy = ann["translation"][:2]
    d = dist_xy(ego_xy, obj_xy)

    if d < 30:  # only print nearby
        print(f"{category:30s} | {d:.2f} m")