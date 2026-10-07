import hashlib
import json
from pathlib import Path

out=Path('/root/autodl-tmp/pi05_eval/step999_100seeds_v1')
protocol=json.loads((out/'protocol.json').read_text())
summary=json.loads((out/'summary.json').read_text())
rows=[json.loads(p.read_text()) for p in sorted((out/'episodes').glob('*.json'))]
assert len(rows)==100 and {r['seed'] for r in rows}==set(protocol['seeds'])
manifest=json.loads((out/'checkpoint_sha256.json').read_text())
for rel,expected in manifest.items():
    path=out/'checkpoint'/rel;h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert path.stat().st_size==expected['size'] and h.hexdigest()==expected['sha256'],rel
summary['checkpoint_hash_unchanged']=True
summary['videos_ok']=all(r.get('video_exit_code')==0 for r in rows if r['valid_scene'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
def percent(x):return 'N/A' if x is None else f'{100*x:.1f}%'
def seconds(x):return 'N/A（无成功样本）' if x is None else f'{x:.3f} s'
ci=summary['success_rate_wilson_95']
lines=['# π0.5 固定权重闭环评测报告','',
    f"模型：`{summary['model']}`；固定训练 checkpoint：`999`（已训练 1,000 步）。",
    '', '## 总体结果', '', '| 指标 | 结果 |', '|---|---|',
    f"| 随机种子数 | {summary['completed']} / 100 |",
    f"| 有效初始化场景 | {summary['valid_scenes']} |",
    f"| Success | {summary['successes']} / 100 |",
    f"| Success Rate（所有预选种子） | {percent(summary['success_rate'])} |",
    f"| Success Rate 的 95% Wilson 区间 | {percent(ci[0])}–{percent(ci[1])} |",
    f"| Success Rate（有效场景） | {percent(summary['valid_scene_success_rate'])} |",
    f"| Mean Execution Time（所有执行样本） | {seconds(summary['mean_execution_time_s'])} |",
    f"| Median Execution Time（所有执行样本） | {seconds(summary['median_execution_time_s'])} |",
    f"| Mean Time（仅成功样本） | {seconds(summary['mean_success_execution_time_s'])} |",
    f"| Median Time（仅成功样本） | {seconds(summary['median_success_execution_time_s'])} |",
    f"| Mean Simulation Time | {seconds(summary['mean_simulation_time_s'])} |",
    f"| Timeout Rate（所有种子） | {percent(summary['timeout_rate'])} |",
    f"| Collision Rate（所有种子） | {percent(summary['collision_rate'])} |",
    f"| 环境初始化错误 | {summary['setup_errors']} |",
    '', '## Failure Type（互斥的自动分类）', '', '| 类型 | 次数 | 占全部 100 个种子 |', '|---|---|---|']
names={'drop':'Drop（掉落代理判定）','grasp_fail':'Grasp Fail（抓取失败代理判定）',
       'place_fail':'Place Fail（放置失败代理判定）','timeout_unknown':'Timeout / Unknown',
       'environment_setup_error':'环境初始化失败','policy_or_runtime_error':'推理或运行错误'}
for key,name in names.items():
    count=summary['failure_counts'].get(key,0);lines.append(f'| {name} | {count} | {count:.1f}% |')
lines += ['', '## 判定与时间口径', '',
    '- 成功使用原任务 `check_success` / `eval_success`，未修改成功阈值。',
    '- 每次推理最多执行 50 个动作，再读取当前观测。每例上限 700 控制步，另有 300 秒墙钟保护上限。',
    '- 墙钟执行时间排除模型加载、预热、场景初始化及视频最终封装；包含推理、物理执行与逐物理步接触监测。仿真时间单独按物理步数乘 timestep 计算。',
    '- 超时是终止标记，可与抓取失败、掉落或放置失败同时出现，不能把 Timeout Rate 与失败分类比例相加。',
    '- 碰撞率是发生预定义不期望接触的回合比例，不是碰撞导致失败的因果比例。接触冲量阈值 0.001 N·s；排除初始静态接触、相邻连杆及正常夹爪—面包接触。',
    '- 抓取和掉落类别是自动启发式：抬升至少 4 cm 且近期存在指尖接触视为抓取；抓起后回落到原始高度附近且在篮外持续 0.1 仿真秒，或掉到桌面下方，视为掉落。需要时应结合视频人工复核。',
    '- 100 个种子预先确定，不按结果筛选或替换；初始化失败保留在主分母中，有效场景成功率另列。',
    '- 图像视频按每 2 个控制步一帧、10 fps 播放，视频时长不是执行耗时。',
    '- 完整定义、种子、源码快照见 `protocol.json` 与 `environment_source/`。逐回合数据见 `episodes.csv` 与 `episodes/`，视频见 `videos/`。',
    '', '## 完整性', '',
    '- 独立 checkpoint 副本在评测前后逐文件 SHA256 一致；评测程序只有推理调用，无优化器或训练更新。',
    f"- 所有有效回合视频写入正常：{summary['videos_ok']}。",
    '- 本结果衡量当前 checkpoint 在这一固定任务配置下的表现，不代表其他场景或实机成功率。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
(out/'COMPLETE').write_text('100 seeds evaluated; checkpoint hashes verified; report generated.\n')
print('EVALUATION_COMPLETE',out,flush=True)
