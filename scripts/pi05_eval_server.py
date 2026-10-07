"""Frozen-parameter Pi05 server. Reset the sampling RNG for reproducible episodes."""
import argparse
import asyncio
import jax
from XPolicyLab.policy.Pi_05.model import Model
from client_server.ws.model_server import PolicyServer, PolicyServerConfig

class EvaluationModel(Model):
    def reset(self):
        super().reset()
        self.policy._rng=jax.random.key(0)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--port',type=int,default=18086)
    a=p.parse_args()
    model=EvaluationModel(dict(task_name='place_bread_basket',env_cfg_type='aloha_agilex',
        action_type='joint',model_path=a.checkpoint,train_config_name='pi05_bread_lora_4090',
        repo_id='bread_square_100_demo'))
    model.reset()
    asyncio.run(PolicyServer(model,PolicyServerConfig(host='127.0.0.1',port=a.port,ws_ping_timeout_s=None)).serve_forever())
