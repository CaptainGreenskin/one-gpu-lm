# 002 · 从原始字节训练 GPT

用 256 个字节值作为词表，把下一个字节的交叉熵换算为 bits/byte。模型包含学习得到的位置向量、因果自注意力、前馈层和共享输入/输出权重。验证使用固定随机种子抽取的 8 个小批次。这个估计用于排查训练流程，正式对照应对完整留出集逐段评分。

## CPU 自检

在仓库根目录运行。需要 Python 和 PyTorch；已用于本机验证的 CPU 版本为 2.5.1。两份短文是原创玩具语料，分别用于训练和验证。训练 12 步只验证程序能完成参数更新、评估和保存。

```bash
D=experiments/002-first-train
OMP_NUM_THREADS=1 python "$D/train.py" --train "$D/toy-train.txt" --val "$D/toy-val.txt" \
  --device cpu --tiny --steps 12 --run-dir runs/002-cpu
```

## RTX PRO 6000 首轮

在 GPU 主机使用支持 Blackwell 和 BF16 的 PyTorch CUDA 构建。准备有使用权的 UTF-8 纯文本，按源文档分开训练和验证集，去重并记录语料版本。程序将两个文件的 SHA-256 写入报告；调用方负责检查跨集近似重复。先跑上面的玩具命令，把 `--device cpu` 改成 `--device cuda` 并选一个新目录。再进行以下 100 步校准，将环境变量指向实际文本：

```bash
D=experiments/002-first-train
# TRAIN_TXT 和 VAL_TXT 应指向已按源文档划分的文本文件
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "$D/train.py" --train "${TRAIN_TXT:?}" --val "${VAL_TXT:?}" \
  --device cuda --steps 100 --run-dir runs/002-gpu
```

默认模型为 12 层、宽度 768、序列长度 512 字节，每步训练 4,096 字节，100 步为 409,600 字节。每个运行目录保存 `metrics.json` 和最终模型 `model.pt`，要求目录名尚未使用。训练计时覆盖更新循环，排除初始和结束验证。峰值指 PyTorch 已分配显存。不同机器的训练时间需要分开报告。

首轮 GPU 验收应保存驱动记录、运行命令、代码提交、环境与指标，随后在更大语料上延长预算。按需复现实验时请使用独立目录。CPU 自检报告为 [cpu-smoke.json](cpu-smoke.json)。
