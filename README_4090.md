# 4090 ACT 使用记录

项目：`/root/autodl-tmp/RoboTwin`，分支：`robot-cooking-act`。

```bash
cd /root/autodl-tmp/RoboTwin
source scripts/activate_autodl.sh
NUM_EPOCHS=10 BATCH_SIZE=2 bash scripts/train_act_minimal.sh
```

ACT 的 `TASK_CONFIGS.json` 已在 2026-10-07 强制纳入子模块版本管理；`processed_data/` 仍为可重新生成的数据，不由 Git 保存。完整迁移见 `MIGRATION_README.md`。当前已使用仓库原有转换器生成配置和预处理数据，保留 `--is_sim` 对齐方式、三路 RGB 图像和原有 640×480 分辨率。

- 原始 100 条轨迹：`data/bread_square_100/place_bread_basket/aloha_agilex/`
- 小 demo：从上述数据的前 20 条派生，来源记录见 `data/bread_square_20/place_bread_basket/aloha_agilex/subset_manifest.json`。这不是从旧机器单独恢复的原始 20 条数据集。
- ACT 配置：`XPolicyLab/policy/ACT/TASK_CONFIGS.json`
- ACT 预处理数据：`XPolicyLab/policy/ACT/processed_data/bread_square_20/place_bread_basket/aloha_agilex-joint/`

以后如需重建小 demo 数据和配置：

```bash
bash scripts/prepare_act_minimal.sh
```

该命令会重新生成派生数据，预计需要约 13.1 GB 空间，不修改原始 100 条轨迹。

2026-10-06 修复验证：原训练脚本使用 `NUM_EPOCHS=1 BATCH_SIZE=2` 完成训练和验证，退出码 0。结果在 `XPolicyLab/policy/ACT/checkpoints/restore_check_4090_01/`，包含 `policy_last.ckpt`、`dataset_stats.pkl`、`run_config.json`、`metrics.json`。这是流程检查模型，未评估任务成功率。

JSON 检查结果保存在 `/root/autodl-tmp/restore_3080/json_audit.json`。旧 checkpoint 未备份，其历史训练记录无法从 Git 或轨迹包还原。
