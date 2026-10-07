# π0.5 固定 checkpoint 闭环评测

训练检查通过：原始 100 条专家轨迹、1,000 个训练步，最终 checkpoint 为 `999`。本次日志前 50 步平均 loss 为 0.213862，末 50 步为 0.019644，最后一步为 0.0213。未记录独立验证集 loss，因此任务表现由闭环评测确定。

## 运行状态与结果

评测运行在服务器 tmux 的 `pi05_eval:eval100_isolated` 窗口：

```bash
tmux attach -t pi05_eval
# 切到 eval100_isolated 窗口，或直接查看日志：
tail -f /root/autodl-tmp/pi05_eval/step999_100seeds_v1/evaluation.log
```

结果目录：

```text
/root/autodl-tmp/pi05_eval/step999_100seeds_v1/
  training_audit.json       训练检查
  protocol.json             固定协议及全部 100 个种子
  checkpoint_sha256.json    固定权重文件校验值
  checkpoint/               独立复制的权重及归一化统计
  episodes/                 逐种子结果、接触与物体运动事件
  episodes.csv              每回合可直接分析的数据表
  summary.json              每回合自动更新的汇总
  videos/                   每个有效回合的视频
  rollout_logs/             每回合控制步日志
  REPORT.md                 全部完成并通过权重校验后生成的中文报告
  COMPLETE                  全部 100 个种子完成的标记
```

`summary.json` 的 `complete` 只有在全部 100 个种子完成时才为 true。不要把中途汇总当作最终成功率。基础设施检查结果保存在 `pilot_baseline`、`pilot_render_check` 与 `pilot_resource_check`，不计入任何正式指标。

## 固定条件

- 模型：π0.5 LoRA，checkpoint 999；仅推理，没有优化器更新。
- 场景：`place_bread_basket`，`bread_square_20` 配置，`BREAD_DATA_VARIANT=default`。
- 用固定随机数生成器预选 100 个不重复种子，范围 10,000–999,999，与原训练种子范围分离。不会根据结果筛掉困难种子。
- 每次推理执行最多 50 个动作后重新观测；沿用任务的 700 控制步上限，另设 300 秒墙钟保护上限。
- 不采用小 demo 的关节增量限幅；夹爪限制在 [0,1]。保留原仿真、动作控制与成功判定。
- 每个回合重置策略随机数状态；相机观测实时来自当前仿真。
- 单回合串行执行，每回合使用独立仿真子进程并共用固定模型服务；进程启动时间不计入执行耗时。

## 指标解释

输出包括 Success Rate、Mean/Median Execution Time、Failure Type、Timeout Rate、Collision Rate，并另外输出成功样本耗时与仿真时间。

墙钟执行耗时排除加载、预热、场景初始化与视频最终封装，包含推理、执行和接触监测。仿真时间通过物理步数乘 timestep 计算。视频以 10 fps 播放，视频时长不用于计算执行耗时。

碰撞是逐物理步检测到预定义的不期望接触事件，阈值为合计冲量 >0.001 N·s；排除初始静态接触、相邻连杆和正常指尖—面包接触。

失败分类使用接触和物体位姿的**启发式代理判定**，不是人工确认的因果结论：

- Grasp Fail：未建立“抬升至少 4 cm 且近期有指尖接触”的抓取代理状态，物体也未放入篮中。
- Drop：物体落到桌面下方，或抓起后在篮外回落并持续 0.1 仿真秒。
- Place Fail：存在抓取代理状态，但未达到放置成功条件。
- 环境初始化失败与推理/运行错误独立记录，不伪装成抓取失败。

Timeout 是终止标记，与上述分类可能重叠，比例不能直接相加。Collision Rate 是事件比例，也不等于“由碰撞导致失败”的比例。

主要成功率使用全部 100 个预选种子为分母，初始化失败保留在分母并单独报告；另提供有效场景成功率。模型没有成功案例时，“成功样本平均/中位耗时”应为 N/A，不能填 0。

## 中断后继续

仅当原评测进程已经结束、端口 18086 已释放时：

```bash
cd /root/autodl-tmp/RoboTwin
bash scripts/run_pi05_eval.sh
```

脚本会跳过已经保存的回合，继续剩余种子。若出现评测程序错误，需要先检查对应 JSON 与 traceback，不应直接把错误当作有效策略结果。评测结束会自动校验 checkpoint 的 SHA256，并生成最终报告。
