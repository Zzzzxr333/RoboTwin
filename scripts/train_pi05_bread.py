"""Single-4090 pi05 LoRA training on the user's expert bread trajectories."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / 'XPolicyLab/policy/Pi_05/openpi'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch-size must be positive')
    from openpi.training import config
    base = config.get_config('pi05_bread_lora_4090')
    model = base.model
    cfg = dataclasses.replace(base, exp_name=args.run_dir.name,
        batch_size=args.batch_size, num_workers=0, fsdp_devices=1, ema_decay=None,
        num_train_steps=args.steps, log_interval=1, save_interval=1000, keep_period=None,
        checkpoint_dir_override=str(args.run_dir.resolve()), resume=args.resume,
        overwrite=False, wandb_enabled=False)
    spec = importlib.util.spec_from_file_location('openpi_train_entry', OPENPI / 'scripts/train.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    print(f'Training pi05 LoRA: steps={args.steps}, batch={args.batch_size}, run={args.run_dir}', flush=True)
    entry.main(cfg)
    (args.run_dir / 'run_description.json').write_text(json.dumps({
        'model': 'pi05', 'mode': 'LoRA', 'paligemma_variant': model.paligemma_variant,
        'action_expert_variant': model.action_expert_variant,
        'dataset': 'local/bread_square_100', 'total_target_steps': args.steps,
        'batch_size': args.batch_size, 'resumed': args.resume,
        'norm_stats_asset_id': 'bread_square_100_demo', 'entrypoint': str(Path(__file__).resolve()),
    }, indent=2))
    print('TRAINING_COMPLETE', args.run_dir, flush=True)


if __name__ == '__main__':
    main()
