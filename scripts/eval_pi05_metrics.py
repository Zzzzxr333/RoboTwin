"""Fixed-checkpoint RoboTwin rollouts and explicit, reproducible metric definitions."""
import argparse
from collections import Counter
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import eval_policy_xpolicylab as bridge
bridge.add_xpolicylab_paths(ROOT/'XPolicyLab')
from client_server.ws import WsModelClient

class RolloutTimeout(Exception): pass

class Monitor:
    def __init__(self,task):
        self.task=task;self.real=task.scene;self.dt=float(self.real.get_timestep());self.steps=0
        self.deadline=math.inf;self.collisions={};self.max_impulse=0.
        self.initial=np.array([b.get_pose().p for b in task.bread])
        self.ids={b.actor.per_scene_id:i for i,b in enumerate(task.bread)}
        self.recent=np.full(len(task.bread),-1000000);self.grasped=np.zeros(len(task.bread),bool)
        self.dropped=np.zeros(len(task.bread),bool);self.low=np.zeros(len(task.bread),int)
        self.max_lift=np.zeros(len(task.bread));self.events=[]
        self.links={link.entity.per_scene_id:link.name for link in task.robot.left_entity.get_links()}
        self.active={i for i,n in self.links.items() if n.startswith(('fl_link','fr_link')) and n[-1:] in '12345678'}
        self.fingers={i for i,n in self.links.items() if n in ('fl_link7','fl_link8','fr_link7','fr_link8')}
        self.adjacent=set()
        for joint in ET.parse(task.robot.left_urdf_path).getroot().findall('joint'):
            self.adjacent.add(frozenset((joint.find('parent').get('link'),joint.find('child').get('link'))))
        self.baseline={frozenset(b.entity.per_scene_id for b in c.bodies) for c in self.real.get_contacts()}
    def __getattr__(self,name):return getattr(self.real,name)
    def in_basket(self,p):
        return bool(np.all(np.abs(p[:2]-self.task.breadbasket.get_pose().p[:2])<.05) and p[2]>.73+self.task.table_z_bias)
    def step(self):
        if time.monotonic()>self.deadline:raise RolloutTimeout('wall_time_limit')
        self.real.step();self.steps+=1
        for contact in self.real.get_contacts():
            a,b=contact.bodies; ia,ib=a.entity.per_scene_id,b.entity.per_scene_id
            # Static/world/object-only contacts cannot contribute to either metric.
            if not ({ia,ib}&self.active):continue
            intended=(ia in self.fingers and ib in self.ids) or (ib in self.fingers and ia in self.ids)
            pair=frozenset((ia,ib));names=(a.entity.name,b.entity.name)
            ignored=(pair in self.baseline or frozenset(names) in self.adjacent or
                     (ia in self.fingers and ib in self.fingers and names[0][:2]==names[1][:2]))
            if ignored and not intended:continue
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
            for finger,obj in ((ia,ib),(ib,ia)):
                if finger in self.fingers and obj in self.ids and impulse>1e-7:
                    self.recent[self.ids[obj]]=self.steps
            if ignored or intended or impulse<=.001:continue
            label=' | '.join(sorted(names))
            self.collisions[label]=max(impulse,self.collisions.get(label,0.))
            self.max_impulse=max(self.max_impulse,impulse)
        for i,bread in enumerate(self.task.bread):
            p=np.asarray(bread.get_pose().p);lift=float(p[2]-self.initial[i,2])
            self.max_lift[i]=max(self.max_lift[i],lift)
            if lift>=.04 and (self.steps-self.recent[i])*self.dt<=.1 and not self.grasped[i]:
                self.grasped[i]=True;self.events.append({'event':'grasp_proxy','object':i,'sim_s':self.steps*self.dt})
            low=self.grasped[i] and lift<.015 and not self.in_basket(p)
            self.low[i]=self.low[i]+1 if low else 0
            fell=p[2]<(.74+self.task.table_z_bias-.08)
            if (fell or self.low[i]*self.dt>=.1) and not self.dropped[i]:
                self.dropped[i]=True;self.events.append({'event':'drop_proxy','object':i,'sim_s':self.steps*self.dt})

def client(port):
    c=WsModelClient(url=f'ws://127.0.0.1:{port}',evaluation_id='pi05-step999-eval100',
        trial_id='rollout',action_case_id='bread-case',request_timeout_s=900,ws_ping_timeout_s=None)
    c._robotwin_protocol='ws';return c

def setup(seed):
    args,_=bridge.load_task_args(dict(task_name='place_bread_basket',task_config='bread_square_20',policy_name='Pi_05'))
    args.update(eval_mode=True,collect_data=False,eval_video_log=False,render_freq=0)
    args.pop('eval_video_save_dir',None)
    task=bridge.class_decorator('place_bread_basket')
    try:
        task.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
        if not task.bread:raise ValueError('No bread objects generated')
        task.set_instruction('Place the bread into the basket.')
        return task
    except BaseException:
        bridge.safe_close_env(task);raise

def infer(c,task,obs):
    c.call(func_name='update_obs',obs=bridge.robotwin_obs_to_xpolicylab(obs,
        instruction=task.get_instruction(),task_env=task))
    actions=bridge.normalize_action_chunk(c.call(func_name='get_action'))
    if not actions:raise ValueError('Model returned no actions')
    return actions

def run_one(c,seed,index,protocol,out):
    row=dict(index=index,seed=seed,success=False,valid_scene=False,timeout=False,collision=False,
        failure_type=None,termination=None,execution_time_s=None,simulation_time_s=None,control_steps=0,
        inference_calls=0,inference_time_s=0.,video=f'videos/{index:03d}_{seed}.mp4')
    task=None;monitor=None;video=None;start=None;video_error=None
    try:
        try:task=setup(seed)
        except Exception as exc:
            row.update(failure_type='environment_setup_error',termination='setup_error',error=f'{type(exc).__name__}: {exc}')
            return row
        row['valid_scene']=True;row['bread_count']=len(task.bread)
        row['bread_model_ids']=[int(x) for x in task.bread_id];row['basket_model_id']=int(task.basket_id)
        task.step_lim=protocol['max_control_steps'];monitor=Monitor(task);task.scene=monitor
        row['initial_bread_positions']=monitor.initial.tolist()
        bridge.prepare_policy_case(c,'place_bread_basket',seed,task.get_instruction(),'joint')
        bridge.reset_policy(c)
        obs=task.get_obs();rgb=obs['observation']['head_camera']['rgb'];h,w=rgb.shape[:2]
        video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{w}x{h}','-framerate','10','-i','-','-an','-vcodec','libx264','-preset','veryfast',
            '-crf','25','-pix_fmt','yuv420p',str(out/row['video'])],stdin=subprocess.PIPE)
        video.stdin.write(np.ascontiguousarray(rgb,dtype=np.uint8).tobytes())
        start=time.monotonic();monitor.deadline=start+protocol['wall_timeout_s']
        while task.take_action_cnt<task.step_lim and not task.eval_success:
            if time.monotonic()>monitor.deadline:raise RolloutTimeout('wall_time_limit')
            obs=task.get_obs();before=time.monotonic();actions=infer(c,task,obs)
            row['inference_time_s']+=time.monotonic()-before;row['inference_calls']+=1
            for action in actions[:protocol['action_chunk_execute']]:
                flat,kind=bridge.xpolicylab_action_to_robotwin(action,action_type='joint',current_observation=obs)
                flat=np.asarray(flat,dtype=np.float64)
                if flat.shape!=(14,) or not np.isfinite(flat).all():raise ValueError('Invalid policy action')
                flat[[6,13]]=np.clip(flat[[6,13]],0,1)
                task.take_action(flat,action_type=kind)
                row['control_steps']=task.take_action_cnt
                if task.take_action_cnt%2==0 or task.eval_success:
                    frame=task.get_obs()['observation']['head_camera']['rgb']
                    video.stdin.write(np.ascontiguousarray(frame,dtype=np.uint8).tobytes())
                if task.eval_success or task.take_action_cnt>=task.step_lim:break
            print(f"chunk_done steps={task.take_action_cnt} wall_s={time.monotonic()-start:.2f}",flush=True)
        row['success']=bool(task.eval_success)
        row['timeout']=not row['success']
        row['termination']='success' if row['success'] else 'step_limit'
    except RolloutTimeout as exc:
        row.update(timeout=True,termination=str(exc))
    except Exception as exc:
        row.update(failure_type='policy_or_runtime_error',termination='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if start is not None:
            row['execution_time_s']=time.monotonic()-start
            row['control_steps']=int(task.take_action_cnt)
        if monitor is not None:
            row.update(simulation_time_s=monitor.steps*monitor.dt,physics_steps=monitor.steps,
                collision=bool(monitor.collisions),collision_pairs=monitor.collisions,
                max_collision_impulse=monitor.max_impulse,grasped_proxy=monitor.grasped.tolist(),
                dropped_proxy=monitor.dropped.tolist(),max_lift_m=monitor.max_lift.tolist(),events=monitor.events)
            final=np.array([b.get_pose().p for b in task.bread]);placed=[monitor.in_basket(p) for p in final]
            row.update(final_bread_positions=final.tolist(),placed_proxy=placed)
            if row['success']:row['failure_type']='success'
            elif row['failure_type'] is None:
                if monitor.dropped.any():row['failure_type']='drop'
                elif any(not g and not p for g,p in zip(monitor.grasped,placed)):row['failure_type']='grasp_fail'
                elif monitor.grasped.any():row['failure_type']='place_fail'
                else:row['failure_type']='timeout_unknown'
            task.scene=monitor.real
        if video:
            try:video.stdin.close();rc=video.wait(timeout=30)
            except Exception as exc:video_error=str(exc);video.kill();rc=video.wait()
            row['video_exit_code']=rc
            if video_error:row['video_error']=video_error
        if task:bridge.safe_close_env(task,clear_cache=True)
    return row

def aggregate(out,protocol):
    rows=[json.loads(p.read_text()) for p in sorted((out/'episodes').glob('*.json'))]
    n=len(rows);valid=[r for r in rows if r['valid_scene']];success=[r for r in rows if r['success']]
    times=[r['execution_time_s'] for r in valid if r['execution_time_s'] is not None]
    st=[r['execution_time_s'] for r in success]
    sim_times=[r['simulation_time_s'] for r in valid if r['simulation_time_s'] is not None]
    counts=Counter(r['failure_type'] for r in rows)
    z=1.95996398454
    rate=len(success)/n if n else 0.
    center=(rate+z*z/(2*n))/(1+z*z/n) if n else None
    half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n) if n else None
    summary=dict(model=protocol['model'],checkpoint=protocol['checkpoint'],planned_seeds=len(protocol['seeds']),
        completed=n,valid_scenes=len(valid),successes=len(success),success_rate=len(success)/n if n else None,
        valid_scene_success_rate=len(success)/len(valid) if valid else None,
        mean_execution_time_s=statistics.mean(times) if times else None,
        median_execution_time_s=statistics.median(times) if times else None,
        mean_success_execution_time_s=statistics.mean(st) if st else None,
        median_success_execution_time_s=statistics.median(st) if st else None,
        mean_simulation_time_s=statistics.mean(sim_times) if sim_times else None,
        timeout_rate=sum(r['timeout'] for r in rows)/n if n else None,
        collision_rate=sum(r['collision'] for r in rows)/n if n else None,
        success_rate_wilson_95=[center-half,center+half] if n else None,
        valid_scene_timeout_rate=sum(r['timeout'] for r in valid)/len(valid) if valid else None,
        valid_scene_collision_rate=sum(r['collision'] for r in valid)/len(valid) if valid else None,
        setup_errors=sum(r['failure_type']=='environment_setup_error' for r in rows),
        failure_counts=dict(counts),failure_rates={k:v/n for k,v in counts.items() if k!='success'},
        complete=n==len(protocol['seeds']),classification='heuristic with per-physics-step contact and object telemetry; see protocol.json')
    temp=out/'summary.json.tmp';temp.write_text(json.dumps(summary,indent=2));temp.replace(out/'summary.json')
    fields=['index','seed','success','valid_scene','execution_time_s','simulation_time_s','timeout','collision','failure_type','termination','control_steps','inference_calls','inference_time_s','video']
    with (out/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18086);p.add_argument('--limit',type=int,default=100)
    p.add_argument('--index',type=int);p.add_argument('--warmup-only',action='store_true');p.add_argument('--skip-warmup',action='store_true')
    a=p.parse_args();os.chdir(ROOT);os.environ['BREAD_DATA_VARIANT']='default'
    protocol=json.loads((a.output/'protocol.json').read_text())
    for name in ['episodes','videos','rollout_logs']:(a.output/name).mkdir(exist_ok=True)
    c=client(a.port)
    try:
        if not a.skip_warmup:
            warm=setup(protocol['warmup_seed'])
            try:
                bridge.reset_policy(c);actions=infer(c,warm,warm.get_obs())
                flat,_=bridge.xpolicylab_action_to_robotwin(actions[0],action_type='joint',current_observation=warm.get_obs())
                assert np.isfinite(flat).all();print('WARMUP_OK',len(actions),flush=True)
            finally:bridge.safe_close_env(warm,clear_cache=True)
        if a.warmup_only:return
        indexes=[a.index] if a.index is not None else range(min(a.limit,len(protocol['seeds'])))
        for index in indexes:
            seed=protocol['seeds'][index]
            path=a.output/'episodes'/f'{index:03d}_{seed}.json'
            if path.exists():continue
            with (a.output/'rollout_logs'/f'{index:03d}_{seed}.log').open('w') as log,contextlib.redirect_stdout(log):
                row=run_one(c,seed,index,protocol,a.output)
            path.write_text(json.dumps(row,indent=2));summary=aggregate(a.output,protocol)
            print(f"EVAL {summary['completed']}/100 seed={seed} success={row['success']} failure={row['failure_type']} time={row['execution_time_s']} collision={row['collision']}",flush=True)
            if row['failure_type']=='policy_or_runtime_error':
                raise RuntimeError('Rollout runtime error; inspect episode JSON before continuing')
        print('EVALUATION_BATCH_FINISHED',json.dumps(aggregate(a.output,protocol)),flush=True)
    finally:bridge.close_policy_client(c)

if __name__=='__main__':main()
