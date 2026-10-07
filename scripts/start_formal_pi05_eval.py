import hashlib
import json
from pathlib import Path
import shutil
out=Path('/root/autodl-tmp/pi05_eval/step999_100seeds_v1')
pilot=out/'pilot_resource_check';pilot.mkdir(exist_ok=True)
for folder in ['episodes','videos','rollout_logs']:
    (pilot/folder).mkdir(exist_ok=True)
    for path in (out/folder).iterdir():
        if path.is_file():shutil.move(str(path),str(pilot/folder/path.name))
for name in ['summary.json','episodes.csv','evaluation.log','policy.log']:
    if (out/name).exists():shutil.move(str(out/name),str(pilot/name))
shutil.copy2(out/'protocol.json',pilot/'protocol.json')
p=json.loads((out/'protocol.json').read_text())
p['render_cadence']='Original RoboTwin rendering calls at every physics substep; no render suppression.'
p['parallel_rollouts']=1
p['process_isolation']='One fresh simulator subprocess per seed; a shared frozen policy server; serial execution.'
p['pilots']='Infrastructure checks archived separately and excluded from all reported metrics; all 100 original seeds rerun.'
p['evaluation_code_sha256']={}
root=Path('/root/autodl-tmp/RoboTwin')
for name in ['eval_pi05_metrics.py','eval_pi05_batch.py','pi05_eval_server.py','run_pi05_eval.sh','finalize_pi05_eval.py']:
    src=root/'scripts'/name;dst=out/'environment_source/scripts'/name;dst.parent.mkdir(exist_ok=True,parents=True)
    shutil.copy2(src,dst);p['evaluation_code_sha256'][name]=hashlib.sha256(src.read_bytes()).hexdigest()
(out/'protocol.json').write_text(json.dumps(p,indent=2))
