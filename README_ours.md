# RoboTwin 环境配置记录

配置日期：2026-10-04。仅完成环境配置与基础检查，未运行任务、采集数据或训练。

## 使用方法

VS Code SSH 项目目录：`/root/autodl-tmp/RoboTwin`。该工作区的新终端默认使用 `RoboTwin (AutoDL)` 配置，自动激活环境。已有终端执行：

```bash
cd /root/autodl-tmp/RoboTwin
source scripts/activate_autodl.sh
```

Python 解释器：`/root/miniconda3/envs/RoboTwin/bin/python`。

## 已安装并验证

- Ubuntu 22.04，RTX 3080 Ti 12GB，驱动 595.71.05。
- Python 3.10.8；CUDA Toolkit 12.1；PyTorch 2.4.1+cu121；torchvision 0.19.1+cu121。
- NumPy 1.26.4；SAPIEN 3.0.0b1；MPlib 0.2.1；PyTorch3D 0.7.8。
- CuRobo 官方 v0.7.8；Warp 1.12.0；XPolicyLab；FFmpeg；Vulkan。
- `pip check`：`No broken requirements found`。
- CUDA 实际张量运算通过，识别 GPU 为 RTX 3080 Ti。
- PyTorch3D 编译扩展、CuRobo MotionGen、RoboTwin CuroboPlanner 导入通过。
- 官方 `python scripts/test_render.py` 返回 `Render Well`。这验证渲染初始化，不代表任务或采集已验收。

`nvidia-smi` 显示的 CUDA 13.2 是驱动支持上限；实际 Toolkit 和 PyTorch CUDA 均为 12.1。

## 代码版本

上游：<https://github.com/RoboTwin-Platform/RoboTwin>

- RoboTwin：`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`
- XPolicyLab：`0ccd8e9f3ed76c1a3186a5b151bead5b53cf2249`
- CuRobo v0.7.8：`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- 工作分支：`cooking/day1-setup`；本次配置文件保留在工作区，未提交或推送。

官方安装脚本将 XPolicyLab 更新到上述提交，因此仓库显示子模块改动是预期状态。

## 资产范围

官方机器人资产 `embodiments` 和物体资产 `objects` 已完整下载、通过 ZIP 校验并解压。已生成当前服务器所需的 6 个机器人路径配置。

**约 11GB 的随机背景纹理包尚未下载。** 干净场景配置不使用它；运行启用 `random_background` 的配置前，需要补齐官方 `background_texture.zip` 并解压到 `assets/`。

之前准备的 `day1_smoke.yml`（1 条）和 `day1_small.yml`（5 条）仅是未执行的配置文件。所有 Demo、采集、训练及评测均由用户后续自行开展。

## 安装处理

- 使用 AutoDL 学术加速下载 GitHub / Hugging Face 资源；子模块下载断流后重试成功。
- 独立 Conda 环境从 base 克隆，并按官方脚本升级 RoboTwin 所需依赖，没有修改 base 环境。
- 固定 PyTorch/torchvision 的 cu121 构建与 NumPy 1.26.4，执行官方 `scripts/_install.sh`。
- 安装 Vulkan、FFmpeg、编译工具等系统依赖；启动脚本指定 NVIDIA Vulkan ICD 和 CUDA 路径。
- 将独立环境中的 TensorBoard 升级到 2.21.0，修复旧版与 protobuf 的冲突。
- 当前仍可能看到 SAPIEN 弃用提示、Requests 字符编码依赖提示；上述基础检查均已通过，没有为消除所有 warning 而继续扩大修改。

## 文件与日志

- 服务器安装日志：`/root/autodl-tmp/setup-logs/`
- 启动脚本：`scripts/activate_autodl.sh`
- VS Code 终端初始化：`scripts/autodl_terminal.sh`
- VS Code 设置：`.vscode/settings.json`
- 完整包版本与 Conda 导出保存在随附的 `robotwin-environment-records.zip`。

环境配置完成不等于实验完成。按用户要求，没有用任务运行、HDF5 产出或训练结果作为本次交付结论。
