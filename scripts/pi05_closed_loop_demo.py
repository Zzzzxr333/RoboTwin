"""Real pi05 websocket inference in RoboTwin, with bounded receding-horizon execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_policy_xpolicylab as bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18085)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--chunk', type=int, default=5)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--max-joint-step', type=float, default=.08)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.rounds < 2 or not 1 <= options.chunk <= 50 or options.max_joint_step <= 0:
        parser.error('Require rounds >= 2, 1 <= chunk <= 50, and positive max-joint-step')
    os.chdir(ROOT)
    options.output.mkdir(parents=True, exist_ok=True)
    settings = dict(task_name='place_bread_basket', task_config='bread_square_20',
                    policy_name='Pi_05', ckpt_name='pi05_base_workflow_demo',
                    host='127.0.0.1', port=options.port, protocol='ws')
    args, _ = bridge.load_task_args(settings)
    args.update(eval_mode=True, render_freq=0, collect_data=False, eval_video_log=False)
    args.pop('eval_video_save_dir', None)
    instruction = 'Place the bread into the basket.'
    task = bridge.class_decorator(settings['task_name'])
    client = None
    video = None
    report = dict(model='official pi05_base', fine_tuned=False,
                  normalization='bread_square_100_demo', seed=options.seed,
                  instruction=instruction, requested_rounds=options.rounds,
                  executed_steps=0, rounds=[], pipeline_passed=False, task_success=False,
                  max_joint_step=options.max_joint_step,
                  note='Workflow demonstration; task success is not expected from unfine-tuned weights.')
    try:
        task.setup_demo(now_ep_num=0, seed=options.seed, is_test=True, **args)
        task.step_lim = options.rounds * options.chunk
        task.set_instruction(instruction)
        bridge.add_xpolicylab_paths(ROOT / 'XPolicyLab')
        from client_server.ws import WsModelClient
        client = WsModelClient(url=f'ws://127.0.0.1:{options.port}',
                               evaluation_id='pi05-demo', trial_id=f'demo-{options.seed}',
                               action_case_id='bread-demo', request_timeout_s=900,
                               ws_ping_timeout_s=None)
        client._robotwin_protocol = 'ws'
        bridge.prepare_policy_case(client, settings['task_name'], options.seed, instruction, 'joint')
        bridge.reset_policy(client)

        def frame(obs):
            nonlocal video
            rgb = np.ascontiguousarray(obs['observation']['head_camera']['rgb'], dtype=np.uint8)
            if video is None:
                height, width = rgb.shape[:2]
                video = subprocess.Popen([
                    'ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                    '-video_size', f'{width}x{height}', '-framerate', '10', '-i', '-',
                    '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                    str(options.output / 'demo.mp4')], stdin=subprocess.PIPE)
            video.stdin.write(rgb.tobytes())
            return hashlib.sha256(rgb.tobytes()).hexdigest()

        for round_index in range(options.rounds):
            observation = task.get_obs()
            image_hash = frame(observation)
            packed = bridge.robotwin_obs_to_xpolicylab(observation, instruction=instruction, task_env=task)
            before = time.monotonic()
            client.call(func_name='update_obs', obs=packed)
            actions = bridge.normalize_action_chunk(client.call(func_name='get_action'))
            latency = time.monotonic() - before
            if not actions:
                raise RuntimeError('Empty model action chunk')
            row = dict(round=round_index + 1, inference_seconds=latency,
                       returned_actions=len(actions), observation_sha256=image_hash,
                       steps=0, clipped_components=0)
            for action in actions[:options.chunk]:
                flat, kind = bridge.xpolicylab_action_to_robotwin(
                    action, action_type='joint', current_observation=observation)
                flat = np.asarray(flat, dtype=np.float64)
                if flat.shape != (14,) or not np.isfinite(flat).all():
                    raise ValueError(f'Invalid pi05 output: {flat}')
                current = np.concatenate([
                    bridge.current_joint_arm(observation, 'left'),
                    [bridge.current_gripper(observation, 'left')],
                    bridge.current_joint_arm(observation, 'right'),
                    [bridge.current_gripper(observation, 'right')]])
                if row['steps'] == 0:
                    row['input_joint_state'] = current.tolist()
                    row['raw_first_action'] = flat.tolist()
                bounded = np.clip(flat, current - options.max_joint_step, current + options.max_joint_step)
                bounded[[6, 13]] = np.clip(flat[[6, 13]], 0, 1)
                row['clipped_components'] += int(np.count_nonzero(bounded != flat))
                task.take_action(bounded, action_type=kind)
                report['executed_steps'] += 1
                row['steps'] += 1
                observation = task.get_obs()
                frame(observation)
                if task.eval_success:
                    break
            report['rounds'].append(row)
            print(f"Round {row['round']}: {latency:.2f}s inference, {row['steps']} actions executed", flush=True)
            if task.eval_success:
                break
        report['task_success'] = bool(task.eval_success)
        report['pipeline_passed'] = len(report['rounds']) >= 2 and report['executed_steps'] > 0
        if not report['pipeline_passed']:
            raise RuntimeError('Fewer than two feedback rounds completed; closed-loop check is incomplete')
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if video:
            video.stdin.close()
            report['video_exit_code'] = video.wait()
        (options.output / 'summary.json').write_text(json.dumps(report, indent=2))
        if client:
            bridge.close_policy_client(client)
        bridge.safe_close_env(task)
    if report.get('video_exit_code') != 0:
        raise RuntimeError('Video encoder did not complete successfully')
    print('CLOSED_LOOP_PASS:', options.output, flush=True)


if __name__ == '__main__':
    main()
