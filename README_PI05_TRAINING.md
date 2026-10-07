# π0.5 单张 4090 训练

## 实测结果（2026-10-07）

已完成 5 步真实训练、保存 checkpoint，再恢复参数与优化器状态续训 5 步，累计 10 步。10 步的 loss 和梯度均为有限值，最后一步 loss 为 0.1056。最新 checkpoint 为 `/opt/pi05/checkpoints/bread_lora_4090/9`（目录编号从 0 开始，对应已完成 10 步）。π0.5 与 RoboTwin 两个环境均通过依赖检查。

完整指标与日志路径保存在 `/root/autodl-tmp/pi05-demo-logs/training_verification_20261007.json`。这是训练环境验证，尚未评估任务成功率。

## 启动训练

在服务器 SSH 终端运行：

```bash
cd /root/autodl-tmp/RoboTwin
STEPS=1000 bash scripts/train_pi05_bread.sh
```

默认 batch size 为 1，使用原始 `bread_square_100` 的 100 条专家轨迹（23,604 帧）。采用 **LoRA 微调**，只更新 49,987,584 个 LoRA 参数，其余参数冻结。这是单张 24 GB 4090 的配置，不能将其理解为全参数训练。默认继续已验证的训练运行，训练到总计 1,000 步；新目录才会从官方 π0.5 base 权重开始。

建议放进 tmux：

```bash
tmux new -s pi05_train
cd /root/autodl-tmp/RoboTwin
STEPS=1000 bash scripts/train_pi05_bread.sh
```

按 `Ctrl+B` 再按 `D` 可离开 tmux 而保持训练；重新进入用 `tmux attach -t pi05_train`。训练期间应保持 GPU 空闲，不要同时运行数据采集或另一个模型服务。

脚本会打印 checkpoint 目录。默认目录为：

```text
/opt/pi05/checkpoints/bread_lora_4090/
```

日志在 `/opt/pi05/logs/`。`STEPS` 是目标训练总步数，不是 epoch 数。短训练仅用于验证链路，不能据此判断放面包任务的成功率。

## 续训

指定已有运行目录，目标总步数必须大于已经完成的步数：

```bash
RUN_DIR=/opt/pi05/checkpoints/你的运行目录 RESUME=1 STEPS=2000 \
  bash scripts/train_pi05_bread.sh
```

默认每 1,000 步及结束时保存，保留最近一份 checkpoint。保存新 checkpoint 时需要同时容纳旧、新两份，请留意系统盘空间。目录已存在时默认续训；`RESUME=0` 可强制要求从头开始，此时若目录已存在会报错，不会覆盖旧结果。

如需独立的新实验，先确认磁盘容量，再通过 `RUN_DIR=/opt/pi05/checkpoints/新实验名称` 指定新目录。

## 已配置内容

- Python：`/opt/pi05/.venv/bin/python`，Python 3.11。
- JAX 0.5.3、PyTorch 2.10.0、LeRobot 0.4.4。
- π0.5 环境统一到 NumPy 2.2.6，以满足 LeRobot/Rerun；原 RoboTwin 仿真环境仍独立。
- 在 OpenPI 与 openpi-client 的依赖声明中记录 NumPy 版本，重新生成锁文件；`pytest`、`h5py`、`msgpack-numpy`、XPolicyLab 已纳入必需依赖。
- 锁文件使用清华 PyPI 镜像。若要同步该环境，在 OpenPI 目录运行 `uv sync --group lerobot --no-dev --frozen`；日常训练直接使用上面的启动脚本即可。
- 原依赖声明备份在 `/root/autodl-tmp/pi05-demo-logs/before_training_setup/`。
- 官方基础模型：`/root/autodl-tmp/checkpoints/pi05_base`。29 个文件已校验大小及 CRC32C。
- 训练配置名：`pi05_bread_lora_4090`，已注册到 OpenPI，可用于后续加载对应 LoRA checkpoint。
- 数据读取缓存放在 `/root/autodl-tmp/pi05_data/hf_cache`，编译缓存放在 `/opt/pi05/jax_cache`，避免数据缓存挤占系统盘的 checkpoint 空间。

## 数据

转换后的 LeRobot v3 数据：

```text
/root/autodl-tmp/pi05_data/local/bread_square_100/
```

三路图像为 224×224 RGB，状态和动作均为 14 维，动作来自原始 HDF5 的 `action` 字段。转换没有用下一帧状态代替动作。图像通过 XPolicyLab 共享解码函数解码。

使用名义 30 Hz 时间戳组织帧，不做重采样；训练按帧索引取 50 帧动作窗口。归一化统计来自相同 100 条数据，使用关节增量和绝对夹爪值。语言指令固定为 `Place the bread into the basket.`。

数据转换脚本：`scripts/prepare_pi05_training_data.py`。完整转换记录为数据目录下的 `conversion_manifest.json`，包含来源、episode 数与总帧数。已有完整数据会复用；发现未完成目录会报错，不会删除原始数据。

## 保存配置

本次既新增了 RoboTwin 下的训练脚本，也修改了 XPolicyLab 子模块中的依赖与训练配置。备份或推送时需要同时保存两部分。`/opt/pi05` 位于系统盘，需要随实例镜像保存；模型、数据盘目录需要另外备份。

官方硬件说明：https://github.com/Physical-Intelligence/openpi#requirements
