# 002 · 从原始字节训练 GPT

用 256 个字节值作为词表，把下一个字节的交叉熵换算为 bits/byte。模型包含学习得到的位置向量、因果自注意力、前馈层和共享输入/输出权重。验证使用固定随机种子抽取的 8 个小批次。这个估计用于排查训练流程，正式对照应对完整留出集逐段评分。

## CPU 自检

在仓库根目录运行。需要 Python 和 PyTorch；已用于本机验证的 CPU 版本为 2.5.1。两份短文是原创玩具语料，分别用于训练和验证。训练 12 步只验证程序能完成参数更新、评估和保存。

```bash
D=experiments/002-first-train
OMP_NUM_THREADS=1 python "$D/train.py" --train "$D/toy-train.txt" --val "$D/toy-val.txt" \
  --device cpu --tiny --steps 12 --run-dir runs/002-cpu
```

## Linux 工作站上的首轮 GPU 训练

使用单张 RTX PRO 6000 Blackwell 96 GB，Python 3.10–3.12，Linux x86_64。请在装卡的机器上运行。带数据的移交包解压后，仓库、Git 提交、语料和文章来源都在同一个目录里；模型从随机权重开始训练。

```bash
tar -xzf one-gpu-lm-first-gpu.tar.gz
cd one-gpu-lm
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
CUDA_VISIBLE_DEVICES=0 bash scripts/first_gpu.sh
```

若工作站使用多张显卡，将 0 换成目标卡的索引。仅有 Git 源码时，先按下文重建语料。移交包含真实语料，因此训练所需的 Python 第三方依赖只有 PyTorch 及其安装依赖。PyTorch 2.8.0 + cu128 是本次固定的构建；RTX PRO 6000 Blackwell 的计算能力为 12.0。驱动需要能运行 CUDA 12.8。启动器会实际运行 BF16 因果注意力的反向传播，检验驱动、软件构建和选中的卡能否协同工作。[PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/) · [NVIDIA 计算能力表](https://developer.nvidia.com/cuda/gpus)

| 阶段 | 做什么 | 留下什么 |
| --- | --- | --- |
| 1 | 校验语料哈希，记录驱动、设备与库版本，再运行 BF16 内核 | hardware.csv、environment.json、pip-freeze.txt、data-manifest.json |
| 2 | 在小语料上跑 12 步 GPU 自检 | smoke/metrics.json 和小权重 |
| 3 | 在中文维基语料上从零训练 100 步 | train/metrics.json、train/model.pt |
| 4 | 核对数据、步数和有效指标，在 CPU 上读回最终 checkpoint | summary.json 和终端 PASS；完整过程见 console.log |

每次生成独立的 `runs/pro6000-wiki-…/`，失败后修好环境可重跑。模型有 85,645,824 个参数，采用 12 层、宽度 768、12 个头、上下文 512 **字节**、batch 8、AdamW（lr=3e-4）、seed 42。矩阵运算可使用 BF16，模型参数和优化器状态使用 FP32。这版一个 UTF-8 字节就是一个训练 token。每步处理 4,096 个字节，100 步共 409,600 个字节。随机切片可以从文章中间开始，也可以跨越两个换行组成的文章边界。

观察每秒训练字节数、PyTorch 峰值已分配显存和训练前后验证 bits/byte。吞吐只覆盖训练更新循环，包含首步开销；显存数与 nvidia-smi 的整卡占用定义不同。验证固定抽取 8 个批次，本轮用于建立机器和程序的基线。将这份 summary 和运行元数据带回项目，再据吞吐计划下一阶段的语料量和卡时。

**如果启动失败。** 若 `nvidia-smi` 查不到卡，先检查主机驱动或容器 GPU 映射。若显示 CUDA 不可用或内核不支持 sm_120，核对当前虚拟环境、cu128 wheel 和驱动，并重新运行自检。若显存不足，先在 `nvidia-smi` 确认选中的卡和可用显存，再以相同参数重跑。如主动更改 batch 或上下文，请另记一次实验。GPU 上的真实性能以实际工作站记录为准。

## 本次中文语料

选用 Wikimedia 的 2023-11-01 中文维基百科快照 `20231101.zh`，固定仓库提交 `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`。它提供页面 ID、URL、标题和正文，适合按文章划分并回溯来源。为控制首次下载量，取单一分片 `train-00002-of-00006.parquet`，共 126,796,358 字节。这是一份百科体裁的小样本。[固定版本数据卡与许可](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) · [原始分片](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.zh/train-00002-of-00006.parquet)

程序保留规范化后正文大小为 1,024–32,768 字节的页面，按页面 ID 及合并空白后的相同正文去重。对 ID 做带固定前缀的 SHA-256 排序，前 256 篇作验证，随后 2,048 篇作训练。未做近重复检测或跨页面改写检测。文章原始繁简写法保留。训练集 7,669,687 字节，验证集 986,543 字节（含文章分隔符）。逐文件大小和哈希见 [版本清单](wiki-zh-reference.json)。`articles.jsonl` 保存标题、原文链接和切分，`ATTRIBUTION.txt` 保存作者群体、许可和处理说明。数据卡列出 CC BY-SA 3.0 和 GFDL，并说明部分文本可能仅适用 CC 许可。移交包保留这些资料，正文按 CC BY-SA 3.0 分享。[数据许可](https://creativecommons.org/licenses/by-sa/3.0/)

若从源码或空目录重建，在仓库根目录运行。PyArrow 只在构建数据时使用：

```bash
python -m pip install pyarrow==20.0.0
python scripts/wiki_zh.py prepare
python scripts/wiki_zh.py verify
```

命令校验原始分片和生成文件，已有正确输出时会直接复用。也可用 `prepare --shard /path/to/train-00002-of-00006.parquet` 读取已有分片。原文、模型权重和运行目录均由 Git 忽略。

原始玩具语料的 CPU 验证见 [cpu-smoke.json](cpu-smoke.json)。
