# 007 · 换一份中文技术语料，BPE 会在第几遍开始过拟合？

**实测结果。** Vue 的三个种子分别从第 9、10、10 遍出现持续分叉，平均曲线从第 9 遍开始，与 Rust 的平均起点相同。[曲线、逐种子明细和跨语料对照](results/2026-11-05/)

复验对象是 Vue 官方中文指南和 API 文档。固定上游版本，保留这两个栏目中至少 512 字节的 Markdown 页面，按栏目分别以相对路径的 SHA-256 排序，序号从零算起，每四篇取第一篇留作验证。风格指南的候选页面为英文，未纳入。按完整文件划分后有 60 篇训练文档（679,992 字节）和 20 篇验证文档（187,227 字节）。Vue 每遍比 006 的 Rust 文本少 4.9% 原文字节，但同一词表产生 404,892 个 BPE token，比 Rust 的 356,079 个多 13.7%；每遍分别更新 226、238 步。保留 Markdown、代码示例和标记，统一 UTF-8 文本的换行、NFC 和文件首尾空白。语料署名 Yuxi (Evan) You 与 Vue 文档贡献者，所选文档采用 CC BY 4.0。下载脚本保留上游许可。[上游固定版本](https://github.com/vuejs-translations/docs-zh-cn/tree/9f4cd576000707fe65ec600fb154724951dcdfd0) · [许可](https://github.com/vuejs-translations/docs-zh-cn/blob/9f4cd576000707fe65ec600fb154724951dcdfd0/LICENSE) · [文件划分与 SHA-256](source-files.json)

训练前固定判据。种子 42、43、44 各从零训练 24 遍，沿用 006 的 85.6M 参数 GPT、冻结的 2,048 项 BPE 词表、AdamW 学习率 0.0003、每批至多 8 个片段、384 原文字节上下文。每遍末用同一组权重评估完整训练与验证集。先算每遍的三种子平均，再找验证最低点，取最低点后首次连续两遍“验证比前一遍变差、训练比前一遍改善”的第一遍作为过拟合起点。最低点、首次回升、持续分叉起点分开报告，逐种子也使用同一规则。主比较是 Vue 平均曲线的持续分叉起点是否落在第 8～10 遍，并与 Rust 的均值起点第 9 遍对照。不同语料的绝对分数和每遍更新数单独列示。[006 的规则和原始曲线](../006-bpe-epoch-curve/)

## 用自己的文档跑

在仓库根目录运行。准备两个互不包含的目录，每篇完整文档作为一个 UTF-8 `.md` 或 `.txt` 文件，可嵌套子目录。把同一篇文档的不同版本和译本放在同一侧。程序拒绝重复的完整文档，并从训练集移除与验证集完全相同的 384 字节边界片段。其余近重复内容需要由提供文档的人检查。

在安装了 PyTorch 的环境中安装编码和绘图库。本次正式训练使用 Linux、PyTorch 2.14.1+cu130、单张 RTX PRO 6000 Blackwell。其他 CUDA 显卡可运行同一入口，程序先在可见卡上实际测试 BF16 注意力的反向传播。[本项目 GPU 安装示例](../002-first-train/README.md#linux-工作站上的首轮-gpu-训练)

```bash
python -m pip install tokenizers==0.21.4 "matplotlib>=3.6"
exp=experiments/007-document-curves
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python "$exp/run.py" \
  --train-dir ./my-docs/train --val-dir ./my-docs/val --out runs/my-docs
python "$exp/report.py" runs/my-docs --out runs/my-docs/report --title 'My documents'
```

输出包括 `report/learning-curve.png`（平均训练与验证曲线、各个种子的验证曲线）、`report/epochs.csv`（每遍分数、训练字节、用时和顺序哈希）和 `report/summary.json`（最低点、首次回升与持续分叉）。这些拐点由整条曲线回看得到。实际部署早停规则时，须预先规定如何仅根据当时可见的验证分数停止，并留出独立的测试文档验证选择。每次训练使用新的输出目录。默认预算为三种子各 24 遍。只想在笔记本检查输入与出图时，可取几篇文档，在训练命令末尾加 `--tiny-cpu --epochs 3 --seeds 42`。该选项使用更小的 CPU 模型，报告命令相同。

## 复现 Vue 实验

```bash
exp=experiments/007-document-curves
python "$exp/fetch_vue.py" --out data/vue-007
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python "$exp/run.py" \
  --train-dir data/vue-007/train --val-dir data/vue-007/val \
  --reference "$exp/prepared-reference.json" --out runs/vue-007
python "$exp/report.py" runs/vue-007 --out runs/vue-007/report --title 'Vue Chinese docs'
```

若只想复核已公开的结果，无需 GPU、语料和权重。安装 Matplotlib 后运行：

```bash
exp=experiments/007-document-curves
python "$exp/report.py" "$exp/results/2026-11-05" --out runs/vue-replot --title 'Vue Chinese docs'
```

## 如何贡献一次复跑

可以在训练仓库开一个 issue，写下硬件、语料种类、文档划分、每遍字节数和验证最低点，附上走势图或逐遍分数。若据此改变了训练时长，也记录原计划与实际选了哪一遍。公开图和汇总只需要指标。原文、文件名清单和权重可由运行者保留在自己的机器上。
