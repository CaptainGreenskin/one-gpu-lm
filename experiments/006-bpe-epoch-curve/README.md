# 006 · 中文 BPE 从第几遍开始过拟合？

**实测结果。** 三个种子的平均验证分数在第 8 遍最低，从第 9 遍起回升。逐种子最低点分别在第 10、8、8 遍。[训练与验证曲线、原始数据](results/2026-11-05/)

005 中，两组端点显示 BPE 训练 4 遍的 Rust 验证分数优于 24 遍。本次对种子 42、43、44 各从零连续训练 24 遍，每遍结束时冻结当前参数，分别在**全部训练文本**和**全部留出文本**上评分。每条轨迹的各个点共享前面的训练历史。每遍看同一批训练片段，随机重排一次，每步最多 8 个片段，238 步为一遍。

沿用 005 的数据清单、冻结的 2,048 项维基 BPE 词表、GPT、AdamW（学习率 0.0003）和 BF16。训练集有 715,084 个原文字节，验证集是第 04、10、15、20 章的 267,047 字节。两个分数都用规范 BPE 序列的总负对数似然除以各自的原文字节数，以 2 为底，越低越好。训练分数用遍末权重重新测量，与验证使用相同的评估模式。[005 数据与配对结果](../005-technical-bytes/)

**预先约定的判读方法。** 对第 1 至 24 遍逐点取三个种子的算术均值，同时保留逐种子曲线。报告验证均值最低的遍数（并列取最早）和之后首次回升的遍数。为区分单点波动，从最低点之后找首次连续两遍均满足“验证分数较上一遍升高、训练分数较上一遍降低”的区间，报告这两遍中的第一遍。每个种子也独立采用同一规则。若无满足条件的区间，记为未观察到。本实验描述这个固定验证集上的过拟合迹象，遍数精度为一遍。第 4 和 24 遍与 005 按种子核对采样顺序哈希，并列出重新训练的验证分数差。

在仓库根目录、使用与 005 相同的 GPU 环境运行。若已有 005 的数据缓存，可从第二条命令开始。评估耗时从训练计时中扣除，整个运行需要另外预留评估时间。

```bash
python experiments/005-technical-bytes/prepare.py
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python experiments/006-bpe-epoch-curve/run.py --out runs/006-bpe-epoch-curve
python -m pip install "matplotlib>=3.6"
python experiments/006-bpe-epoch-curve/report.py runs/006-bpe-epoch-curve --out runs/006-report
```

只想复核曲线时，下载本仓库后直接用公开的每遍记录生成图，无需 GPU 或训练权重：

```bash
python -m pip install "matplotlib>=3.6"
exp=experiments/006-bpe-epoch-curve
python "$exp/report.py" "$exp/results/2026-11-05" --out runs/006-replot
```

若要在自己的语料上找停止时机，先按完整文档分出训练集和验证集。可沿用 [005 的数据准备示例](../005-technical-bytes/prepare.py) 和 [004 的分块与编码函数](../004-tokenizer-soak/prepare.py) 构建数据目录，然后用通用训练入口记录一条曲线：

```bash
python experiments/004-tokenizer-soak/trial.py --arm bpe --seed 42 --epochs 24 --epoch-curve \
  --data "$MY_ENCODED_DATA" --out runs/my-bpe-42
```

`curve.json` 的 `train_bpb` 和 `val_bpb` 分别是整份训练集和验证集的遍末分数。更换种子重复后，可比较各自的最低点与分叉区间。正式选定停止时机后，另留独立测试文档检验最终模型。
