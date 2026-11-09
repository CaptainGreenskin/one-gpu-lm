# 010 · 7B 上下文长度与重算的成本

009 在长度 128、batch 1 上验证了单卡 7B 的真实参数更新。010 把长度扩展为 512、1024、2048、4096，回答两个实际配置问题：可用显存能支持多长的训练序列？按 Transformer block 重算能省多少显存，吞吐代价是多少？

每个长度测全参数 BF16 + Adafactor、冻结 BF16 主干 + QKV LoRA rank 8 / AdamW 两种模式。每种模式对照开启、关闭 activation checkpointing。采用 PyTorch [非重入 checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html)；按 block 重算前向来减少保存的激活，保留 RNG 状态。LoRA 输入本身无需梯度，也必须能回传到适配器参数。

**预先固定的口径。** [protocol.json](protocol.json) 固定 48 个独立进程：4 个长度 × 2 种更新方式 × 2 种重算设置 × 3 次重复。各测 8 个完整优化器更新，前 2 次热身、后 6 次计吞吐；重复 2 反转模式和重算顺序。相同模型种子和独立数据种子，计时外生成并上传合成 token。计时包括清梯度、前向、FP32 交叉熵、反向、检查梯度覆盖、裁剪和优化器更新，CUDA 同步后结束。初始化与整进程用时另记。重复用于评估短时性能波动，并非独立语料训练种子。

所有长度使用同一模型：32 层、32 头、宽度 4096、MLP 18048、词表 32000、共享输入输出权重、位置表 4096，共 **7,028,273,152 参数**。009 的位置表长度为 384，因此其 7,013,068,800 参数及耗时不能直接作为本实验对照。训练参数、梯度为 BF16，全参数优化器没有 FP32 主权重；LoRA 适配器为 FP32。未启用 CPU 卸载、量化、编译或梯度累积，使用 PyTorch 默认 SDPA 调度。

峰值报告 PyTorch allocated 和 reserved；硬件每 2 秒另采样设备总显存、GPU 利用率、温度、功率和时钟。每个进程的分配器上限取 88 GiB 与当前空闲显存减 4 GiB 的较小值，设备不足时保留 skip/OOM。单项超时 180 秒、整组预算 30 分钟，不影响其他进程。原始 JSON 保留各步损失、梯度范数、参数改动、输入哈希和源码哈希。配对审计要求输入、初始参数采样一致，损失绝对差 ≤ 0.001、梯度范数相对差 ≤ 1%；不符合者不得用于性能结论。

在 [009 的 PyTorch/CUDA 环境](../009-training-to-serving/) 中从仓库根目录执行：

```bash
exp=experiments/010-7b-context-scaling
python -m unittest discover -s "$exp" -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python "$exp/run.py" --out runs/context010
python "$exp/report.py" runs/context010 --out runs/context010-report
```

输出目录必须是新目录，防止覆盖和重复启动。报告依赖 Matplotlib；训练依赖 PyTorch 和 NumPy。仓库只同步源码、配置、指标和曲线，日志及模型留在本机。**随机权重、合成 token 和短测只能证明该配置的容量与性能，不能证明长训收敛、真实语料质量或兼容现成的 Llama/Qwen 权重。**
