import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics

ROOT = Path('/root/autodl-tmp/RoboTwin')
OUT = Path('/root/autodl-tmp/pi05_eval/step999_100seeds_v1')
SOURCE = Path('/opt/pi05/checkpoints/bread_lora_4090/999')
OUT.mkdir(parents=True, exist_ok=True)
log = Path('/opt/pi05/logs/bread_lora_4090_20261007_010746.log').read_text()
assert 'TRAINING_COMPLETE' in log and SOURCE.is_dir()
rows = [(int(s),float(g),float(l)) for s,g,l in re.findall(r'Step (\d+): grad_norm=([\d.e+-]+), loss=([\d.e+-]+)',log)]
assert rows[-1][0] == 999 and all(math.isfinite(g) and math.isfinite(l) for _,g,l in rows)
audit = {'training_steps':1000,'dataset_episodes':100,'logged_steps':len(rows),
         'first_50_mean_loss':statistics.mean(x[2] for x in rows[:50]),
         'last_50_mean_loss':statistics.mean(x[2] for x in rows[-50:]),
         'last_loss':rows[-1][2], 'all_metrics_finite':True,
         'source_checkpoint':str(SOURCE), 'note':'No held-out validation loss was logged; rollout performance remains unverified.'}
(OUT/'training_audit.json').write_text(json.dumps(audit,indent=2))
snapshot=OUT/'checkpoint'
if not snapshot.exists():
    snapshot.mkdir()
    for name in ['params','assets']:
        shutil.copytree(SOURCE/name,snapshot/name)
manifest={}
for path in sorted(snapshot.rglob('*')):
    if path.is_file():
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        manifest[str(path.relative_to(snapshot))]={'size':path.stat().st_size,'sha256':h.hexdigest()}
(OUT/'checkpoint_sha256.json').write_text(json.dumps(manifest,indent=2))
seeds=random.Random(20261007).sample(range(10000,1000000),100)
protocol={'model':'pi05_bread_lora_step999','checkpoint':str(snapshot),'checkpoint_step':999,
    'train_config_name':'pi05_bread_lora_4090','normalization':'bread_square_100_demo',
    'task':'place_bread_basket','task_config':'bread_square_20','variant':'default',
    'seed_generator_seed':20261007,'seeds':seeds,'warmup_seed':100,'policy_rng_seed':0,
    'action_chunk_execute':50,'max_control_steps':700,'wall_timeout_s':300,
    'extra_joint_clipping':False,'gripper_clipping':[0,1],
    'success':'Unmodified task.check_success / task.eval_success',
    'seed_selection':'100 preselected seeds outside original training seed range; no success-based filtering or replacements',
    'timing':'Wall execution excludes model load, warmup, environment setup and video finalization; includes RPC, simulation and frame recording. Simulation time is physics steps * dt.',
    'collision':'At any physics step: non-baseline/non-adjacent contact involving fl/fr moving arm or finger links, excluding intended finger-bread contact; summed impulse norms > 0.001 N*s. Episode event rate, not collision-caused failure rate.',
    'grasp_proxy':'Object raised >=4 cm above initial center with finger contact during previous 0.1 simulation seconds; heuristic, not a grasp sensor.',
    'drop_proxy':'Object falls >=8 cm below table top, or after grasp proxy returns within 1.5 cm of initial height outside basket for >=0.1 simulation seconds.',
    'failure_priority':['environment_setup_error','policy_or_runtime_error','drop','grasp_fail','place_fail','timeout_unknown'],
    'timeout':'Separate termination flag, overlaps failure types. True when unsuccessful rollout reaches 700 actions or 300 wall seconds.',
    'primary_denominator':100,'invalid_seeds':'Included in attempted-seed denominator and reported separately; also report valid-scene success rate.',
    'video':'Head RGB, frame every 2 control steps, 10 fps playback; playback duration is not execution time.',
    'updates':'Inference only; no optimizer or parameter updates; identical per-episode model RNG seed.'}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
for rel in ['envs/place_bread_basket.py','env_cfg/task_config/bread_square_20.yml']:
    dst=OUT/'environment_source'/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/rel,dst)
print(json.dumps(audit,indent=2));print('EVAL_PREPARED',OUT,flush=True)
