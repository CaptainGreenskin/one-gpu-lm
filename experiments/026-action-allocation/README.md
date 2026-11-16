# 026：从学生卡住的状态补充修复示范

**诊断与接手执行器已完成，正在冻结首次 GPU 试跑协议；尚无新的 GPU 结果。** [025](../025-verified-distillation/results/2026-11-15-factorial/)的自蒸馏未达到提升门槛。本阶段先解释可观测行为，再设计有明确边界的训练侧接手试跑，不根据五道开发退步题直接造训练答案。

[完整诊断与图表](results/2026-11-16-diagnosis/)核对全部 164 条采集轨迹和 720 次开发评测，新增模型调用和参数更新均为 0：

- 五道退步题的学生都耗尽 12 次调用而不修改源码；没有触发上下文裁剪或单次生成长度上限。不能据此把问题归咎于上下文容量。
- 971 条实际训练动作中，131 条修改源码，133 条单独回归验证，127 条提交，580 条为其他动作。保守语法分类把其中 420 条识别为查阅形态命令；“其他”不等于无效阅读。
- 127 条合格完整轨迹只有 4 条包含多次源码修改。16 条教师轨迹超过学生开发时的 12 次调用上限。纠错覆盖与预算匹配值得验证，但这些统计尚不能确立训练退步的因果机制。

下一步的[接手试跑设计](recovery-pilot-design.json)已选定 48 道训练题、5 个来源家族。选题只依赖固定哈希和仓库轮转，不按成功或失败挑题；与既有训练数据有重叠，不冒充独立开发集。计划比较学生与底座教师从同一第 3 次反馈状态继续、使用相同剩余预算时的结果。教师只获得当时学生可见的历史和工作目录。

实时状态快照、隔离恢复、配对预算和逐步审计已实现，并通过本地与真实 Linux 沙箱 CPU 检查。[执行协议](execution-protocol.json)固定模型、数据、代码、任务顺序、预算和门槛；公开文件核验和 CI 通过后才允许启动唯一一次 `recovery026-001`。`recovery-pilot-design.json`保留设计说明，启动以执行协议为准。只有采集可行性通过，才进入下一轮候选训练准备；不会因为此诊断直接重训、开启最终测试或上传参数。

文件内容、空目录、权限、mtime、可见历史和已消耗预算会被复制；inode、ctime、atime、外部时钟及进程内存不保留。快照限 4096 个条目、64 MiB，遇到符号链接、硬链接或特殊文件保留学生结果并记为未配对。两分支均使用显式提交验证规则，不能在教师接手时重置 12 次调用预算。首次 CPU 沙箱检查见 [准备记录](preparation/cpu-sandbox-check.json)。

运行与事后审计入口：

```bash
python -m one_gpu_lm.verified.recovery_watch --root /path/to/run-root --protocol /path/to/execution-protocol.json
python -m one_gpu_lm.verified.recovery_audit --root /path/to/run-root --protocol /path/to/execution-protocol.json --out /path/to/new-audit.json
```

该协议引用已有私有运行目录和经审查的任务，不是无需准备数据即可运行的公共题库。真实接手轨迹保持私有，报告发布安全数字、配对表、硬件采样和图形。

## 复用

对保存着 025 私有原始记录的运行目录，只读生成安全数字：

```bash
python -m one_gpu_lm.verified.behavior_diagnosis \
  --root /path/to/original-run-root \
  --out /path/to/new-diagnosis/diagnosis.json
```

它校验原审计轨迹哈希、工具元数据、反馈重算和冻结文件，不执行任务命令。原任务定义只用于取训练/开发任务的可见回归命令，不执行隐藏检查，也不访问既有最终测试。命令语法分类是保守的形态标签，不是 shell 语义证明。

没有私有任务也可以从公开数字复算本报告：

```bash
python -m one_gpu_lm.verified.behavior_report \
  --results experiments/026-action-allocation/results/2026-11-16-diagnosis
```

绘图需要 matplotlib；`--no-figures` 仅使用标准库复算表格和文字。原始命令、任务、模型文本、权重和连接信息不随报告发布。
