# 013：27B 单卡训练基础设施

目标是让同一份合法数据和固定基模能通过标准入口检查、训练、保存完整状态、换进程恢复、回归评测并收集公开证据。主实现位于 `one_gpu_lm`，实验目录只保存配方、提前固定的验收条件和结果。

## 验收设计

- 固定 Qwen/Qwen3.8-27B revision、同一张 RTX PRO 6000 Blackwell 96 GB、BF16 底座与 rank-8 LoRA。
- 沿用 012 的私有训练分区，最多选 16 条，固定顺序更新 12 次；不新增语料或调用教师 API。
- A 组连续 12 步；B 组在第 4 步提交检查点并退出，新进程恢复至第 12 步。
- 比较逐步 loss、adapter、AdamW、随机状态、样本顺序，以及保存/重载的贪心探针。阈值在 [protocol.json](protocol.json) 中固定。
- 对保存的产物运行已有九道函数回归；这是工程 smoke，不作模型质量晋升依据。
- CPU 另验证损坏检查点、不兼容身份、半写入与互斥锁。公开产物通过白名单收集，不含数据、主机信息、权重或优化器文件。

## 使用与结果

安装环境、准备数据、运行训练和恢复的完整命令见 [27B 操作手册](../../docs/27b-training.md)。

```bash
python -m one_gpu_lm verify-resume \
  --config experiments/013-training-infrastructure/recovery-config.json \
  --out runs/recovery-001 --interrupt-after 4
```

已发布的配对、失败分析与硬件记录见 [2026-11-09 结果](results/2026-11-09/)。用 `report.py --results <结果目录>` 重建报告与曲线。外部读者需要用自己有权使用的记录满足数据契约；本公开包不分发 012 的私有逐样本数据。

这是单卡 LoRA 基础设施，未包含全参数 27B、多卡训练、生产服务或新的质量结论。保存机制在本地 POSIX 文件系统上工作，GPU 验收限定完整优化器更新边界的进程中断。
