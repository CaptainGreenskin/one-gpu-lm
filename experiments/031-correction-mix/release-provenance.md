# 031 适配器的来源和许可核验

核验从 031 实际训练文件开始。1,006 条样本的消息、目标、出处和编码回执逐条对回各自的已审计导出。971 条来自完整轨迹，35 条来自纠错后的已执行动作。两批本地模型服务回执都绑定 `Qwen/Qwen3.8-27B` 的固定 revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`。原轨迹采集的控制记录中，外部教师请求计数为零。[031 训练审计](results/2026-11-17-training/integrity.json) · [030 纠错协议](../030-teacher-reasoning/execution-protocol.json)

| 训练来源 | 使用样本 | 保留的许可和署名材料 |
| --- | ---: | --- |
| gweis/isodate | 204 | [BSD-3-Clause 主许可](https://github.com/gweis/isodate/blob/17cb25eb7bc3556a68f3f7b241313e9bb8b23760/LICENSE) · [CPython 许可](https://github.com/python/cpython/blob/2c2462afc5d978d31f3d073001fc202d269917ff/LICENSE) |
| jashkenas/underscore | 322 | [MIT 主许可](https://github.com/jashkenas/underscore/blob/5c1e5f686f8500af9c9d15614ed60289d6d35d6b/LICENSE) · [CPython 许可](https://github.com/python/cpython/blob/2c2462afc5d978d31f3d073001fc202d269917ff/LICENSE) |
| jpvanhal/inflection | 102 | [MIT 主许可](https://github.com/jpvanhal/inflection/blob/88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1/LICENSE) · [Rails ActiveSupport 许可](https://github.com/rails/rails/blob/914caca2d31bd753f47f9168f2a375921d9e91cc/activesupport/MIT-LICENSE) |
| npm/ini | 244 | [ISC 主许可](https://github.com/npm/ini/blob/3c96c74fd42584bd655e17a4e63e2ef0a3b406ee/LICENSE) |
| python-humanize/humanize | 134 | [主许可](https://github.com/python-humanize/humanize/blob/a95004512dde4e8f0269946bb4af754382347441/LICENCE) · [Django BSD 许可](https://github.com/django/django/blob/6a0dc2176f4ebf907e124d433411e52bba39a28e/LICENSE) |

2026 年 11 月 21 日，底座许可、五份主许可和四份附加声明均与上游固定版本逐字节一致。CPython 和 Rails 链接定位本项目保留的许可材料。原移植所用的具体上游版本仍待确认。

底座仓库的固定版本标记为 Apache-2.0，文件目录列有 `LICENSE`。按其分发条款准备许可原文与适用署名，并在模型卡注明底座来源、revision 和 LoRA 修改范围。[Qwen 固定版本许可](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/37561816e66b5b9153b1b64c10968c18c1e6f4a0/LICENSE) · [Apache-2.0 分发条款](https://www.apache.org/licenses/LICENSE-2.0#redistribution)

五个来源的原始审查均覆盖内部训练。原台账的公开参数项仍待分发范围审查。公开发布决策将以实际入选的适配器、训练输入和拟发布文件为范围记录许可依据与适用条件。[ModelScope 发布标准](../../docs/modelscope-release.md)
