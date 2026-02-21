from nuscenes.nuscenes import NuScenes

NUSCENES_ROOT = "data/nuscenes"
VERSION = "v1.0-mini"

nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=True)

print("Total scenes:", len(nusc.scene))
print("First scene:", nusc.scene[0]["name"])