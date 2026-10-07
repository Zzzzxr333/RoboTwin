"""Convert real XPolicyLab episodes to LeRobot v3 without replacing recorded actions."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, pack_robot_state

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {'cam_head': 'cam_high', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', default='local/bread_square_100')
    parser.add_argument('--output', type=Path, default=Path('/root/autodl-tmp/pi05_data/local/bread_square_100'))
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    raw = ROOT / 'data/bread_square_100/place_bread_basket/aloha_agilex/data'
    files = sorted(raw.glob('episode_*.hdf5'))[:args.episodes]
    if len(files) != args.episodes:
        raise ValueError('Missing expert episodes')
    manifest = args.output / 'conversion_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['episodes'] != args.episodes:
            raise ValueError('Existing converted dataset has a different episode count')
        print('Dataset already prepared:', manifest)
        return
    if args.output.exists():
        raise FileExistsError(f'Unfinished/existing output; inspect before replacing: {args.output}')
    info = get_robot_action_dim_info('aloha_agilex')
    first = load_hdf5(str(files[0]))
    dim = pack_robot_state(first, 'joint', info, source_type='dataset', state_type='state').shape[1]
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (dim,), 'names': None},
        'action': {'dtype': 'float32', 'shape': (dim,), 'names': None},
    }
    for camera in CAMERAS.values():
        features[f'observation.images.{camera}'] = {
            'dtype': 'image', 'shape': (3, 224, 224), 'names': ['channels', 'height', 'width']}
    dataset = LeRobotDataset.create(args.repo_id, fps=30, root=args.output,
                                    robot_type='aloha_agilex', features=features,
                                    use_videos=False, image_writer_threads=4)
    count = 0
    for ep_index, path in enumerate(files):
        data = first if ep_index == 0 else load_hdf5(str(path))
        state = pack_robot_state(data, 'joint', info, source_type='dataset', state_type='state').astype(np.float32)
        actions = pack_robot_state(data, 'joint', info, source_type='dataset', state_type='action').astype(np.float32)
        if state.shape != actions.shape or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f'Invalid state/actions in {path}')
        for i in range(len(state)):
            frame = {'observation.state': state[i], 'action': actions[i],
                     'task': 'Place the bread into the basket.'}
            for source, destination in CAMERAS.items():
                rgb = decode_image_bit(data['vision'][source]['colors'][i])
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
                frame[f'observation.images.{destination}'] = np.ascontiguousarray(rgb.transpose(2, 0, 1))
            dataset.add_frame(frame)
        dataset.save_episode()
        count += len(state)
        print(f'Converted {ep_index + 1}/{len(files)} episodes, {count} frames', flush=True)
    dataset.finalize()
    manifest.write_text(json.dumps({'source': str(raw), 'episodes': len(files), 'frames': count,
        'repo_id': args.repo_id, 'format': 'LeRobot v3', 'images': '224x224 RGB',
        'actions': 'original recorded action; no state shifting',
        'fps': 30, 'fps_note': 'Frame-index sampling at a nominal 30 Hz; no temporal resampling.'}, indent=2))
    print('DATASET_COMPLETE', manifest, flush=True)


if __name__ == '__main__':
    main()
