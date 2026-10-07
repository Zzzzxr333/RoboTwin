import json
import shutil
from pathlib import Path
import h5py

root = Path('/root/autodl-tmp/RoboTwin')
source = root/'data/bread_square_100/place_bread_basket/aloha_agilex/data'
target = root/'data/bread_square_20/place_bread_basket/aloha_agilex/data'
files = [source/f'episode_{i:07d}.hdf5' for i in range(20)]
assert all(p.is_file() for p in files)
frames = 0
for p in files:
    with h5py.File(p) as f:
        frames += len(f['state/left_arm_joint_states'])
estimate = frames * 3 * 480 * 640 * 3
assert shutil.disk_usage(root).free > estimate + 2*1024**3, 'Insufficient space for the official ACT converter'
target.mkdir(parents=True, exist_ok=True)
for p in files:
    link = target/p.name
    if link.exists() or link.is_symlink():
        assert link.resolve() == p.resolve(), f'Existing different data: {link}'
    else:
        link.symlink_to(p)
manifest = {'description': 'Demo subset rebuilt from the first 20 of the 100 restored trajectories; not an independent recovered old dataset.',
            'source': str(source), 'episodes': [p.name for p in files],
            'frame_count': frames, 'estimated_processed_bytes': estimate}
(target.parent/'subset_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
