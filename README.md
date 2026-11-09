# One GPU LM

在一台工作站上，把数据准备、训练与验证、模型导出和推理服务串起来，记录模型质量、卡时、吞吐与显存成本。每项改动都提供对照、原始记录和复现入口，让读者用自己的文档或显卡核验结论。

目前包含三条相互补充的实验线：约 85.6M 参数的中文模型用于真实文档训练、早停、独立测试和部署；约 7B 参数的随机模型用于训练系统的容量与性能测量；已有 27B 开源权重用于编码后训练链路验收。随机 7B 使用合成 token，并未作为具有中文能力的预训练模型发布。

下一阶段面向可长期使用的多语言编码模型：在已有开源底座上做后训练，以仓库任务完成率和长任务恢复能力验收，同时保留模型、教师与训练数据的来源和许可记录。[首次真实 27B LoRA 链路](experiments/011-coding-model-selection/) 已完成；小样本试验出现 9/9 → 7/9 的回归，微调版未被推荐或发布为日用模型。

本仓库保存训练程序、实验配置、复现命令和结果。配套知识项目 [InfraGuide](https://github.com/CaptainGreenskin/infra-guide) 讲原理。理解第一个训练脚本时，可以对照它的 [Transformer 模块](https://captaingreenskin.github.io/infra-guide/article/transformer-block/) 和 [自动微分与训练](https://captaingreenskin.github.io/infra-guide/article/autograd-training/)。

**让下一代基模复用成果。** [模型升级协议](docs/model-upgrades.md) 规定新模型如何接入、对照和发布；[training_lab](training_lab/) 把模型 profile 与数据、隔离评测、LoRA 和报告流程分开。GitHub 保存可审计的实现和实验，ModelScope 承载后续通过验收的具体模型版本，InfraGuide 解释原理。当前尚未发布编码模型权重。

**拿自己的文档做实验。** 把 UTF-8 文档按篇放入训练和留出两个目录，即可逐遍比较训练分数和留出分数，找出继续训练何时开始影响未见文档。可用小型 CPU 模型先检查流程，也可用单张支持 BF16 的 CUDA 显卡跑完整配置。[用自己的文档训练和出图](experiments/007-document-curves/#用自己的文档跑)

**首轮实测。** RTX PRO 6000 Blackwell 单卡完成了 100 步训练。8,565 万参数的 GPT 在中文维基验证样本上从 8.106 降到 5.386 bits/byte，训练更新用时 4.376 秒。读者可按相同数据、软件和计时口径复跑。[实测条件与原始结果](experiments/002-first-train/results/pro6000-first/) · [安装与复现](experiments/002-first-train/README.md#linux-工作站上的首轮-gpu-训练)

| 实验 | 能回答什么 | 复现入口 |
| --- | --- | --- |
| 001 · 同一段文字被切成几个 token | 两个公开分词器编码 12 条固定样例时，输入长度有什么不同？ | [实验记录](experiments/001-tokenizer/README.md) |
| 002 · 从零训练一个 GPT | 训练、留出验证、指标记录和保存权重能否整套跑通？ | [CPU 自检和首轮 GPU 命令](experiments/002-first-train/README.md) |
| 003 · 训练步数对照 | 同一条训练轨迹多训练 900 步，完整验证集的 bits/byte 改善多少？ | [实验方法与结果](experiments/003-training-budget/) |
| 004 · 八小时分词对照 | 在同样的单卡时间内，两种表示怎样影响验证分数和原文字节吞吐？ | [实验设计与复现](experiments/004-tokenizer-soak/) |
| 005 · 中文技术语料 | 长预算的维基结果能否迁移到 Rust 技术文本？相同原文字节数下哪组学习效果更好？ | [预注册方案与复现](experiments/005-technical-bytes/) |
| 006 · 中文 BPE 学习曲线 | 逐遍比较训练章节和留出章节，从哪一遍起出现过拟合迹象？ | [实验方法、曲线与结果](experiments/006-bpe-epoch-curve/) |
| 007 · 换文档复验过拟合 | Rust 的拐点能迁移到 Vue 文档吗？自己的文档应该练几遍？ | [两个目录跑出自己的验证曲线](experiments/007-document-curves/) |
| 008 · 在线早停 | 只用当前验证分数何时停机？与固定 8、24 遍比，独立测试质量与卡时如何？ | [规则、曲线与独立测试](experiments/008-online-early-stop/results/2026-11-05/) |
| 009 · 数据到部署 | 训练、早停、模型导出、KV cache 和 HTTP 续写如何串起来？单卡能否实际更新 7B 参数？ | [全链路入口](experiments/009-training-to-serving/) · [吞吐与显存实测](experiments/009-training-to-serving/results/2026-11-05/) |
| 010 · 7B 长序列训练 | 512–4096 token 的全参数与 LoRA 训练需要多少显存？按层重算何时值得开启？ | [复现与数值审计](experiments/010-7b-context-scaling/) · [曲线与原始结果](experiments/010-7b-context-scaling/results/2026-11-09/) |
| 011 · 编码底座与 LoRA 链路 | 真实 27B 能否加载、微调、重载和隔离评测？工程成功是否等于能力提升？ | [固定版本与复用入口](experiments/011-coding-model-selection/) · [9/9 → 7/9 的回归结果](experiments/011-coding-model-selection/results/2026-11-09/) |

**首月验收。** 先用单卡记录 100 步的训练吞吐、峰值显存和验证集 bits/byte（预测每个原始字节所需的平均比特数，越低越好）。再在同一显卡和时间预算上改变一个因素，例如分词器。公开实验要固定训练和验证文档、模型参数预算、上下文策略与计时口径，并保留多个种子的结果。不同词表会改变输入长度、输出层开销和每步读到的原始字节数，应分别记录。

**后续重点。** 先建立真实仓库任务、独立留出和统一工具预算，再用有明确来源与授权的代码轨迹比较底座与后训练版本。确认质量收益后，完善长任务恢复、部署、首 token 延迟和吞吐对照，再考虑向较小模型蒸馏。每次新基模发布都沿用升级协议和回归系统，欢迎外部贡献者提供其他模型、显卡和合法任务的复测。

**每个实验留什么。** 记录假设、数据来源与版本、设备与环境、代码提交、完整配置、命令、原始指标、失败原因及下一步。权重另存并按数据许可提供获取方式。InfraGuide 的 AI Infra 章节提供对应知识点。

设计参考包括 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 的渐进式教学、[nanochat](https://github.com/karpathy/nanochat) 的完整训练流程和 [autoresearch](https://github.com/karpathy/autoresearch) 的固定算时实验。中文从零训练可以与 [MiniMind](https://github.com/jingyaogong/minimind) 交叉学习。

本仓库的原创代码采用 MIT 许可。中文维基百科语料和实验调用的第三方分词器分别遵循各自的许可，文章出处和处理方式随数据准备脚本生成。[代码许可](LICENSE) · [语料来源与许可](experiments/002-first-train/README.md#本次中文语料) · [分词器来源](experiments/001-tokenizer/README.md)
