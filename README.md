# One GPU LM

在一台工作站上，从随机初始化训练一个中文小语言模型。每次只改一件事，再问：同样的卡时，这项改动能让模型在未见文本上预测得更好吗？读者可以沿着 Git 历史看见模型、训练方法和证据一起演进。

本仓库保存训练程序、实验配置、复现命令和结果。配套知识项目 [InfraGuide](https://github.com/CaptainGreenskin/infra-guide) 讲原理。理解第一个训练脚本时，可以对照它的 [Transformer 模块](https://captaingreenskin.github.io/infra-guide/article/transformer-block/) 和 [自动微分与训练](https://captaingreenskin.github.io/infra-guide/article/autograd-training/)。

**首轮实测。** RTX PRO 6000 Blackwell 单卡完成了 100 步训练。8,565 万参数的 GPT 在中文维基验证样本上从 8.106 降到 5.386 bits/byte，训练更新用时 4.376 秒。读者可按相同数据、软件和计时口径复跑。[实测条件与原始结果](experiments/002-first-train/results/pro6000-first/) · [安装与复现](experiments/002-first-train/README.md#linux-工作站上的首轮-gpu-训练)

| 实验 | 能回答什么 | 复现入口 |
| --- | --- | --- |
| 001 · 同一段文字被切成几个 token | 两个公开分词器编码 12 条固定样例时，输入长度有什么不同？ | [实验记录](experiments/001-tokenizer/README.md) |
| 002 · 从零训练一个 GPT | 训练、留出验证、指标记录和保存权重能否整套跑通？ | [CPU 自检和首轮 GPU 命令](experiments/002-first-train/README.md) |
| 003 · 训练步数对照 | 同一条训练轨迹多训练 900 步，完整验证集的 bits/byte 改善多少？ | [实验方法与结果](experiments/003-training-budget/) |
| 004 · 八小时分词对照 | 在同样的单卡时间内，两种表示怎样影响验证分数和原文字节吞吐？ | [实验设计与复现](experiments/004-tokenizer-soak/) |
| 005 · 中文技术语料 | 长预算的维基结果能否迁移到 Rust 技术文本？相同原文字节数下哪组学习效果更好？ | [预注册方案与复现](experiments/005-technical-bytes/) |
| 006 · 中文 BPE 学习曲线 | 逐遍比较训练章节和留出章节，从哪一遍起出现过拟合迹象？ | [实验方法、曲线与结果](experiments/006-bpe-epoch-curve/) |

**首月验收。** 先用单卡记录 100 步的训练吞吐、峰值显存和验证集 bits/byte（预测每个原始字节所需的平均比特数，越低越好）。再在同一显卡和时间预算上改变一个因素，例如分词器。公开实验要固定训练和验证文档、模型参数预算、上下文策略与计时口径，并保留多个种子的结果。不同词表会改变输入长度、输出层开销和每步读到的原始字节数，应分别记录。

**后续一年。** 第 2–3 个月写出可从干净环境复现的中文预训练和监督微调教程。第 4–6 个月试验短预算的训练改进，并将候选配置延长训练复核；若有第二张卡，可以分头重复对照组。第 7–12 个月把同一模型用于推理实验，记录首 token 延迟和吞吐，请外部贡献者在其他显卡上复跑。此前讨论的 Text-to-SQL 可以作为后续的下游任务。

**每个实验留什么。** 记录假设、数据来源与版本、设备与环境、代码提交、完整配置、命令、原始指标、失败原因及下一步。权重另存并按数据许可提供获取方式。InfraGuide 的 AI Infra 章节提供对应知识点。

设计参考包括 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 的渐进式教学、[nanochat](https://github.com/karpathy/nanochat) 的完整训练流程和 [autoresearch](https://github.com/karpathy/autoresearch) 的固定算时实验。中文从零训练可以与 [MiniMind](https://github.com/jingyaogong/minimind) 交叉学习。

本仓库的原创代码采用 MIT 许可。中文维基百科语料和实验调用的第三方分词器分别遵循各自的许可，文章出处和处理方式随数据准备脚本生成。[代码许可](LICENSE) · [语料来源与许可](experiments/002-first-train/README.md#本次中文语料) · [分词器来源](experiments/001-tokenizer/README.md)
