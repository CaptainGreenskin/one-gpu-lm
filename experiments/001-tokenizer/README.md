# 001 · 同一段话需要多少个 token？

对相同的 UTF-8 输入，分别使用 GPT-2 和 Qwen2.5 分词器。12 条原创例句分成中文、中英文技术混写、英文三组，每组 4 句。两份分词器的版本与 SHA-256 均固定；程序只获取分词器文件，不下载模型权重。计数不包括特殊 token。

在仓库根目录运行：

```bash
uv run experiments/001-tokenizer/run.py
```

首次获取资源需要网络，之后可从本地缓存复现。逐条计数和各组原始字节总量见 [result.json](result.json)。这是输入长度实验。两个词表的大小、训练语料和规则都有差异；要检验训练收益，需要固定语料、模型预算和验证集，单独训练对照组。

[GPT-2 分词器](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json) · [Qwen2.5 分词器](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json)

实测的四句中文共 158 个 GPT-2 token、49 个 Qwen2.5 token，技术混写为 133 和 66，英文均为 45。例如“一张显卡，可以训练出怎样的中文模型？”计数为 39 和 12。这 12 条用于验证统计方法，不能代表任一语言的平均水平。
