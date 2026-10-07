# RoboTwin + π0.5 闭环小 demo（4090）

## 直接运行

在这台服务器的 VS Code SSH 终端执行：

```bash
cd /root/autodl-tmp/RoboTwin
bash scripts/run_pi05_demo.sh
```

默认场景为 `place_bread_basket`，沿用 `bread_square_20.yml` 的干净场景和 `aloha_agilex` 双臂。默认推理 6 轮，每轮仅执行预测序列的前 5 个动作，然后读取新的相机和关节状态，再调用模型。

运行结束输出 `CLOSED_LOOP_PASS`，结果保存到：

```text
/root/autodl-tmp/RoboTwin/eval_result/pi05_demo/<运行时间>/
  demo.mp4       仿真画面
  summary.json   逐轮推理耗时、观测哈希、关节状态、动作与任务结果
  policy.log    模型加载及推理日志
  rollout.log   仿真执行日志
```

在 VS Code 中打开该目录即可查看结果。运行采用无窗口渲染，无需桌面或显示器。脚本退出时自动结束自己启动的模型服务。

延长演示：

```bash
ROUNDS=20 CHUNK=5 bash scripts/run_pi05_demo.sh
```

默认端口 `18085`；如被占用，使用 `PORT=18086`。默认场景种子 `100`；可使用 `SEED=...`，但其他种子可能生成不稳定场景。

## 演示范围

- 使用官方 **π0.5 base** 的真实权重和真实推理，不使用随机动作或专家轨迹回放。
- 这是闭环流程验证，**未针对放面包任务微调，不保证抓取或放置成功**。`pipeline_passed` 与 `task_success` 是两个独立结果。
- 用现有 100 条专家轨迹计算了状态和动作的归一化统计；计算统计不等于模型训练。关节动作采用相对当前状态的增量统计，夹爪使用绝对值，预测长度为 50。
- 为控制未微调模型的动作，演示将每次执行的关节目标限制在当前角度 ±0.08 弧度，夹爪限制到 [0, 1]。报告保留原始首个动作和被限幅的分量数。演示采用 6+1+6+1 的 14 维 ALOHA 接口。
- 没有执行完整任务成功率评测，也没有微调 π0.5。

官方模型说明：https://github.com/Physical-Intelligence/openpi

## 环境与文件

| 用途 | 路径 |
| --- | --- |
| 仿真 Python | `/root/autodl-tmp/conda_envs/RoboTwin/bin/python` |
| π0.5 推理 Python | `/opt/pi05/.venv/bin/python` |
| 官方权重 | `/root/autodl-tmp/checkpoints/pi05_base` |
| 本次归一化配置 | `/root/autodl-tmp/checkpoints/pi05_base/assets/bread_square_100_demo/norm_stats.json` |
| 统计来源说明 | 同目录 `provenance.json` |
| 原始 100 条数据 | `/root/autodl-tmp/RoboTwin/data/bread_square_100/place_bread_basket/aloha_agilex` |
| 下载与安装记录 | `/root/autodl-tmp/pi05-demo-logs` |

启动脚本自动选择两个独立解释器，不需要在终端手动切换 π0.5 环境。π0.5 使用 Python 3.11、JAX 0.5.3、PyTorch 2.10.0+cu128；仿真继续使用原有 RoboTwin 环境。

新增代码：

- `scripts/run_pi05_demo.sh`：启动模型服务、等待就绪、运行闭环并清理进程。
- `scripts/pi05_closed_loop_demo.py`：实时观测、RPC 推理、动作执行和视频记录。
- `scripts/prepare_pi05_demo_stats.py`：从原始 100 条轨迹重新生成归一化 JSON，缺失时启动脚本会自动调用。

复用了仓库现有 Pi_05 适配器和 RoboTwin 的观测/动作转换函数。对 vendored OpenPI 的 `src/openpi/training/checkpoints.py` 做了一处导入修复：把仅用于类型注解的训练数据加载模块放入 `TYPE_CHECKING`，避免推理强制加载训练专用 LeRobot/Rerun 依赖。训练环境现已补齐并完成实际训练验证，见 README_PI05_TRAINING.md；NumPy 已统一为 2.2.6。

权重文件未放进 Git；新增脚本和本说明需要随代码保存，避免下次迁移遗漏。系统盘上的 `/opt/pi05` 需随实例镜像保存，数据盘文件需单独备份。
