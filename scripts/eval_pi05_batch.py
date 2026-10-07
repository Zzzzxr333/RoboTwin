"""Serial isolated simulator processes; policy server stays loaded and frozen."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18086);p.add_argument('--limit',type=int,default=100)
a=p.parse_args();protocol=json.loads((a.output/'protocol.json').read_text())
worker=Path(__file__).with_name('eval_pi05_metrics.py')
base=[sys.executable,str(worker),'--output',str(a.output),'--port',str(a.port)]
subprocess.run(base+['--warmup-only'],check=True)
for index,seed in enumerate(protocol['seeds'][:a.limit]):
    if (a.output/'episodes'/f'{index:03d}_{seed}.json').exists():continue
    subprocess.run(base+['--skip-warmup','--index',str(index)],check=True)
print('ALL_REQUESTED_SEEDS_FINISHED',flush=True)
