# 4090 环境、数据与结果迁移（2026-10-07）

## 备份内容与路径

本地目录：`E:\Robot\cook\4090_migration_20261007`。`SHA256SUMS` 校验压缩包与 Git bundle；每个 `*.verified.json` 记录服务器流与本地文件一致的 SHA256。`ARCHIVE_INVENTORY.json` 检查压缩包可完整读取并记录文件数。

| 文件 | 内容 |
|---|---|
| source_and_configs.tar.gz | 完整代码（含 Git 忽略文件、ACT checkpoint、CuRobo/PyTorch3D 源码）、JSON/YAML、恢复日志；Git 元数据单独用 bundle 保存 |
| datasets.tar.gz | 原始 bread_square_100、bread_light_100、20 条子集链接、转换后的 LeRobot 数据与缓存 |
| trained_checkpoints.tar.gz | π0.5 checkpoint 999（含优化器状态，可续训）、训练日志 |
| evaluation.tar.gz | 完整 100 seeds 评测、固定权重副本、视频、逐例 JSON、CSV、报告、协议与基础设施试跑归档 |
| base_weights.tar.gz | π0.5 基础权重、归一化统计、tokenizer |
| assets.tar.gz | 实际使用的 RoboTwin 物体和机器人资产及原始 zip |
| robotwin_environment.tar.gz | Python 3.10 仿真/ACT 完整已安装环境，含 SAPIEN/mplib 现场补丁 |
| pi05_environment.tar.gz | Python 3.11 运行时与 π0.5 完整虚拟环境 |
| cuda_toolkit.tar.gz | CUDA 12.1 工具链（不含宿主机 NVIDIA 驱动） |
| RoboTwin.bundle / XPolicyLab.bundle | 两个仓库的可离线恢复 Git 提交历史 |

## 推荐恢复方式

选择 Ubuntu 22.04、x86_64、NVIDIA GPU，优先相同 4090；原环境驱动 580 系列。保留原绝对路径，以兼容 conda、venv、editable 包及数据软链接。**仅 git clone 不包含权重、数据、仿真资产和已安装二进制环境。**

把整个本地备份目录上传到新机器。准备足够空间：归档上传空间之外，原数据盘展开约 46 GB，系统 `/opt/pi05` 约 15 GB，CUDA 约 4.4 GB。建议系统盘至少 35 GB、数据盘至少 70 GB（若归档同时存放在数据盘，按压缩包总大小额外预留）。

```bash
bash restore_4090_backup.sh /root/autodl-tmp/4090_migration_20261007
cd /root/autodl-tmp/RoboTwin
source scripts/activate_autodl.sh
python scripts/verify_migration.py
python scripts/test_render.py
```

还需检查 `test_render.py` 输出确实为 `Render Well`（旧脚本可能在错误时也返回 0）。然后执行不计分的策略预热，或运行现有闭环 demo 确认相机、CuRobo 与模型链路。不要用 `start_formal_pi05_eval.py` 重置已完成实验目录。已完成结果直接读 `REPORT.md`。

新机器的 NVIDIA 驱动、容器 GPU 挂载及 Vulkan ICD 由租赁平台提供，不能从旧卡复制驱动。若换成其他 GPU 架构，需重新编译 CuRobo/PyTorch3D 扩展并进行渲染/推理测试；已备份源码与 CUDA 工具链。完整归档消除了重新解析依赖造成的版本漂移，但新机器上的 GPU 验证仍必须实际运行。

## 环境为何要分开

- RoboTwin：Python 3.10、NumPy 1.26.4、Torch 2.4.1/cu121、SAPIEN 3.0.0b1。
- π0.5：Python 3.11、NumPy 2.2.6、JAX 0.5.3、Torch 2.10.0/cu128、LeRobot 0.4.4。
- π0.5 启动时清除 `LD_LIBRARY_PATH`，避免误用仿真的 CUDA 12.1 动态库。现有训练/评测脚本已处理。
- `environment/4090_20261007/` 保存 pip/conda/dpkg 版本清单；含 `file://` 和 editable 路径的 freeze 文件是审计记录，不应盲目当作可跨目录安装的 requirements。
- OpenPI 的 `uv.lock` 与两个 `pyproject.toml` 已保存，ACT 的 `TASK_CONFIGS.json` 已强制纳入 Git，不再遗漏。
- ACT 20 条 processed_data 当前未保留在服务器；可由 `bash scripts/prepare_act_minimal.sh` 重新生成，需要约 13 GB 空间。原始数据和配置均已备份。

## 续训

```bash
cd /root/autodl-tmp/RoboTwin
STEPS=2000 bash scripts/train_pi05_bread.sh
```

会自动从 `/opt/pi05/checkpoints/bread_lora_4090/999` 恢复并继续到总计 2000 步。评测目录中独立的固定 checkpoint 不受续训影响。若要新实验，指定新的 `RUN_DIR`，不要覆盖这次结果。

## GitHub

主仓库继续使用 `robot-cooking-act`；XPolicyLab 改动保存到 `migration-4090-20261007`，主仓库通过提交 ID 固定引用。在线获取用 `git clone --recurse-submodules -b robot-cooking-act https://github.com/Zzzzxr333/RoboTwin.git`。准确提交号见本地 `GIT_COMMITS.json`。

完成备份只代表文件已校验保存；新卡尚未提供，因此不能声称已经在新卡实测恢复成功。
