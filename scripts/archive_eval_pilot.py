import json
from pathlib import Path
import shutil
out=Path('/root/autodl-tmp/pi05_eval/step999_100seeds_v1')
pilot=out/'pilot_baseline';pilot.mkdir(exist_ok=True)
for folder in ['episodes','videos','rollout_logs']:
    (pilot/folder).mkdir(exist_ok=True)
    for path in (out/folder).iterdir():
        if path.is_file():shutil.move(str(path),str(pilot/folder/path.name))
for name in ['summary.json','episodes.csv','evaluation.log','policy.log']:
    if (out/name).exists():shutil.move(str(out/name),str(pilot/name))
shutil.copy2(out/'protocol.json',pilot/'protocol.json')
p=json.loads((out/'protocol.json').read_text())
p['render_cadence']='Headless rendering synchronized for each requested observation only; every physics step still executes contact monitoring and success checks.'
(out/'protocol.json').write_text(json.dumps(p,indent=2))
