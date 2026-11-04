# 005 · 中文技术语料上的等字节分词对照

实验 004 的结果发布后执行两个预先约定的检查。第一项直接用独立技术文本评价 004 的 20/40 分钟 checkpoint，检验维基上的优劣是否能迁移。第二项从随机权重开始，在这份技术语料上训练，并让每个配对读过完全相同的原文、顺序和字节数，记录质量与实际卡时。两个问题分别汇总。

**数据。** 取《Rust 程序设计语言》中文翻译仓库固定版本的 94 个章节文件，仅使用顶层 `src/chNN-*.md`，排除英文原书和附录。去除 HTML 注释和 mdBook 包含指令，保留中文正文、行内代码、Markdown 与原有代码块。按章节隔离，第 04、10、15、20 章留作验证。训练集 715,084 个原文字节，验证集 267,047 个原文字节。全部片段不超过 384 个 UTF-8 字节，编码保留完整字符并检查可逆性；准备程序检查相同片段是否跨集出现。已核对分配：75 个训练文件、19 个验证文件。704 个技术验证片段与实验 004 维基训练集及验证集中的完整片段均无逐字节相同项；此检查仅排除同边界的完全重复。一个译本仅代表一个领域，章节间可能有相似表述。[上游固定提交](https://github.com/rust-lang-cn/book-cn/tree/cde74c448e301ce8ac7960a0d3dc879efd83635d) · [上游 MIT 许可](https://github.com/rust-lang-cn/book-cn/blob/cde74c448e301ce8ac7960a0d3dc879efd83635d/LICENSE-MIT) · [数据清单](prepared-reference.json)

**预注册的预算。** 固定种子 42、43、44，冻结实验 004 的维基 BPE 词表、模型结构、学习率和计分方式。迁移评估使用 12 份已有权重，逐一在留出的 Rust 章节上评分。等字节训练依次比较 4 遍和 24 遍技术训练集，每模型各处理 **2,860,336** 和 **17,162,016** 个原文字节。每遍按种子生成排列，不放回采样，每步最多 8 个片段；两种模型独立从零训练。按序列检查采样轨迹哈希、原文字节数和优化器步数。验证分数为规范 token 序列的 bits/原文字节，越低越好，另报训练更新耗时。

```bash
python -m pip install tokenizers==0.21.4
python experiments/005-technical-bytes/prepare.py
# SOAK_RUN 为实验 004 已完成的八小时运行目录，其中保存模型权重。
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python experiments/005-technical-bytes/run.py \
  --soak "$SOAK_RUN" --data data/rust-zh-pairs --out runs/rust-equal-bytes
python experiments/005-technical-bytes/summarize.py runs/rust-equal-bytes
```

启动器要同时确认八小时窗口已结束、原任务状态已完成且相符的结果已经公开到 GitHub。等待期间不启动 GPU 计算。代码使用仓库的 MIT 许可，数据准备时保存上游两份许可和每个章节的源路径及 SHA-256。原始文本、编码缓存和权重留在工作站。

若只复现等原文字节的对照，可以从数据准备步骤直接开始，无需取得 004 的权重。在仓库根目录用下面的命令运行；已有同名目录时请换一个新的实验目录。

```bash
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
out=runs/rust-equal-independent
for epochs in 4 24; do
  for seed in 42 43 44; do
    arms="byte bpe"
    if (( (seed + epochs / 4) % 2 )); then arms="bpe byte"; fi
    for arm in $arms; do
      python experiments/004-tokenizer-soak/trial.py --data data/rust-zh-pairs --arm "$arm" \
        --seed "$seed" --epochs "$epochs" --out "$out/equal_bytes-$epochs-$seed-$arm-a1"
    done
  done
done
python experiments/005-technical-bytes/summarize.py --equal-only "$out"
```

判读时看每个种子的配对差和均值。三个种子都指向同一侧可报告为“一致领先”；若方向不一致，同时列出两侧赢的次数和差值范围。20/40 分钟模型在维基与技术文本上的方向分开列出。
