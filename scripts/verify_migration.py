"""Run on the restored Linux GPU server. Does not train or overwrite results."""
import json, os, pathlib, subprocess, sys
ROOT=pathlib.Path('/root/autodl-tmp/RoboTwin')
os.chdir(ROOT)
required=[ROOT/'XPolicyLab/policy/ACT/TASK_CONFIGS.json',
 ROOT/'env_cfg/robot/_robot_info.json',ROOT/'XPolicyLab/utils/robot/_robot_info.json',
 pathlib.Path('/root/autodl-tmp/pi05_data/local/bread_square_100/meta/info.json'),
 pathlib.Path('/opt/pi05/checkpoints/bread_lora_4090/999/params'),
 pathlib.Path('/root/autodl-tmp/pi05_eval/step999_100seeds_v1/COMPLETE')]
for p in required:
 assert p.exists(),f'Missing: {p}'
 if p.suffix=='.json':json.loads(p.read_text())
for name in ('bread_square_100','bread_light_100'):
 files=list((ROOT/'data'/name).rglob('*.hdf5'))
 assert len(files)==100,(name,len(files))
 print(name,len(files),'episodes')
env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+':'+str(ROOT/'XPolicyLab')
robot='/root/autodl-tmp/conda_envs/RoboTwin/bin/python'
subprocess.run([robot,'-c',"import torch,sapien,mplib,pytorch3d,curobo; assert torch.cuda.is_available(); print('RoboTwin CUDA',torch.cuda.get_device_name())"],env=env,check=True)
subprocess.run([robot,'-c',"import json; p='XPolicyLab/policy/ACT/TASK_CONFIGS.json'; d=json.load(open(p)); assert 'bread_square_20-place_bread_basket-aloha_agilex-joint' in d; print('ACT configuration OK')"],env=env,check=True)
env.pop('LD_LIBRARY_PATH',None);env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
subprocess.run(['/opt/pi05/.venv/bin/python','-c',"import jax; from openpi.training.config import get_config; assert any(x.platform=='gpu' for x in jax.devices()); c=get_config('pi05_bread_lora_4090'); print('pi05 JAX',jax.devices(),c.name)"],env=env,check=True)
print('MIGRATION_IMPORTS_AND_FILES_OK (render and policy warmup must also pass)')
