# 026：从学生卡住的状态补充修复示范

**诊断和首次训练侧接手试跑已完成，未通过预设的扩大采集门槛。** [完整结果、真实曲线与中文复盘](results/2026-11-16-recovery-pilot/discussion.md)：48 道固定训练题全部审计通过，34 组完整配对中学生成功 23 组、底座接手成功 25 组；新增成功 3 题、退步 1 题，净增 2 题，未达到至少 3 题的要求。取得 21 条合格接手轨迹、覆盖 4 个训练来源家族，但每条恰好一次源码修改，仍未补足多次修改的成功纠错示范。没有新增参数更新，最终测试继续封存，未发布权重。

[025](../025-verified-distillation/results/2026-11-15-factorial/)的自蒸馏未达到提升门槛。本阶段先解释可观测行为，再验证有明确边界的训练侧接手采集方法，不根据五道开发退步题直接造训练答案。本次单次 GPU 试跑北京时间 2026-11-16 10:52:57 至 13:25:04，实际 152.12 分钟；687 次实际调用，共享前缀只计一次。硬件记录共 1811 条、采样错误 0；本地只读 SSH 观察连接断开后恢复，远端原任务持续运行，未重启或延长截止时间。

[完整诊断与图表](results/2026-11-16-diagnosis/)核对全部 164 条采集轨迹和 720 次开发评测，新增模型调用和参数更新均为 0：

- 五道退步题的学生都耗尽 12 次调用而不修改源码；没有触发上下文裁剪或单次生成长度上限。不能据此把问题归咎于上下文容量。
- 971 条实际训练动作中，131 条修改源码，133 条单独回归验证，127 条提交，580 条为其他动作。保守语法分类把其中 420 条识别为查阅形态命令；“其他”不等于无效阅读。
- 127 条合格完整轨迹只有 4 条包含多次源码修改。16 条教师轨迹超过学生开发时的 12 次调用上限。纠错覆盖与预算匹配值得验证，但这些统计尚不能确立训练退步的因果机制。

[接手试跑设计](recovery-pilot-design.json)预先选定 48 道训练题、5 个来源家族。选题只依赖固定哈希和仓库轮转，不按成功或失败挑题；与既有训练数据可能重叠，不冒充独立开发集。实验比较学生与底座教师从同一第 3 次反馈状态继续、使用相同剩余预算时的结果。教师只获得当时学生可见的历史和工作目录。另有 14 题在前三次调用内已经修改源码，按固定规则不配对，其学生结果和成本全部保留。

实时状态快照、隔离恢复、配对预算和逐步审计已实现，并通过本地与真实 Linux 沙箱 CPU 检查。[执行协议](execution-protocol.json)固定模型、数据、代码、任务顺序、预算和门槛；公开文件核验和 CI 通过后启动了唯一一次 `recovery026-001`。`recovery-pilot-design.json`保留设计说明，实际执行以执行协议为准。全部请求、反馈、状态、编码、独立判分和物理调用成本均通过事后审计，冻结输入未改变。由于采集可行性未通过，本轮不扩大采集或进入候选训练；后续需先设计能覆盖真实失败后纠错的新训练侧采样规则。

快照复制持久工作目录 `/work` 的文件内容、空目录、权限、mtime，以及可见历史和已消耗预算；inode、ctime、atime、外部时钟及进程内存不保留。只读运行时另行固定，`/tmp`、`/proc`、`/dev` 是每次命令的沙箱状态，不是整个操作系统的快照。快照限 4096 个条目、64 MiB，遇到符号链接、硬链接或特殊文件保留学生结果并记为未配对。两分支均使用显式提交验证规则，不能在教师接手时重置 12 次调用预算。首次 CPU 沙箱检查见 [准备记录](preparation/cpu-sandbox-check.json)。

运行与事后审计入口：

```bash
python -m one_gpu_lm.verified.recovery_watch --root /path/to/run-root --protocol /path/to/execution-protocol.json
python -m one_gpu_lm.verified.recovery_audit --root /path/to/run-root --protocol /path/to/execution-protocol.json --out /path/to/new-audit.json
```

该协议引用已有私有运行目录和经审查的任务，不是无需准备数据即可运行的公共题库。真实接手轨迹保持私有，报告发布安全数字、配对表、硬件采样和图形。

## 复用

无需私有任务，可以从公开安全数字复算接手试跑的表格、硬件统计与曲线（需要 matplotlib）：

```bash
python -m one_gpu_lm.verified.recovery_report \
  --results experiments/026-action-allocation/results/2026-11-16-recovery-pilot
```

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
