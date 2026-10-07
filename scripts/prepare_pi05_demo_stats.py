"""Compute embodiment scaling for a workflow demo; this does NOT train pi05."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state


def read_group(group):
    return {key: read_group(value) if isinstance(value, h5py.Group) else value[()]
            for key, value in group.items()}


def main():
    source = ROOT / 'data/bread_square_100/place_bread_basket/aloha_agilex/data'
    target = Path('/root/autodl-tmp/checkpoints/pi05_base/assets/bread_square_100_demo')
    info = get_robot_action_dim_info('aloha_agilex')
    states, deltas = [], []
    files = sorted(source.glob('episode_*.hdf5'))
    if len(files) != 100:
        raise RuntimeError(f'Expected 100 expert episodes, found {len(files)}')
    for path in files:
        with h5py.File(path, 'r') as file:
            data = {key: read_group(file[key]) for key in ('state', 'action')}
        state = pack_robot_state(data, 'joint', info, source_type='dataset', state_type='state')
        action = pack_robot_state(data, 'joint', info, source_type='dataset', state_type='action')
        if state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f'ALOHA pi05 expects 14 dimensions: {path}, {state.shape}, {action.shape}')
        indexes = np.minimum(np.arange(len(state))[:, None] + np.arange(50), len(state) - 1)
        chunk = action[indexes].copy()
        mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
        chunk[..., mask] -= state[:, None, mask]
        states.append(state)
        deltas.append(chunk.reshape(-1, 14))
    stats = {}
    for key, arrays in [('state', states), ('actions', deltas)]:
        array = np.concatenate(arrays).astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f'Non-finite {key}')
        stats[key] = {name: value.tolist() for name, value in {
            'mean': array.mean(0), 'std': array.std(0),
            'q01': np.quantile(array, .01, axis=0), 'q99': np.quantile(array, .99, axis=0),
        }.items()}
    target.mkdir(parents=True, exist_ok=True)
    (target / 'norm_stats.json').write_text(json.dumps({'norm_stats': stats}, indent=2))
    (target / 'provenance.json').write_text(json.dumps({
        'source': str(source), 'episodes': len(files), 'frames': sum(map(len, states)),
        'horizon': 50, 'adapt_to_pi': False, 'delta_joints': True,
        'grippers': 'absolute', 'trained': False,
        'purpose': 'Workflow smoke demo only; base weights have NOT been fine-tuned on these episodes.',
    }, indent=2))
    print('Saved normalization:', target)


if __name__ == '__main__':
    main()
