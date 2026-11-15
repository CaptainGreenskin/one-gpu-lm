# 新开发题的 CPU 准入

60 道题、6 个仓库家族、两种语言已完成 CPU 准入。此处只有环境与题目检查，**模型评分为 0，教师未接触这些题**。完整四臂评测规则须另行冻结后才能调用模型。

| 仓库 | 题目 | 语言 |
| --- | ---: | --- |
| pallets/markupsafe | 10 | Python |
| pypa/packaging | 10 | Python |
| micromatch/picomatch | 10 | JavaScript |
| tox-dev/platformdirs | 10 | Python |
| npm/node-semver | 10 | JavaScript |
| urllib3/urllib3 | 10 | Python |

准入要求原版通过可见和隐藏检查，故障版分别在两类检查中语义失败，恢复参考源码后通过独立判分。首次通过 56/60，四题的隐藏检查没触及实际故障；保留旧题目版本和首次回执，仅补充独立组合的隐藏用例，第二次通过 60/60。任务描述、故障和可见检查未改变，没有利用模型成绩修题。

[首次准入](development-admission-001.json) · [修订回执](development-task-revision-002.json) · [第二次准入](development-admission-002.json) · [摘要](summary.json)

[来源与许可证清单](sources.json)记录准确版本、源码子集、变更及上游通知。按关联家族排除先前实验仓库；rfc3986、AppDirs 的关联及许可也已保留。运行环境不含宿主包库和参考副本，原始题目、隐藏检查、源码与参考修复不随这些公开回执分发。

这些是构造的模块语义修复题，不是历史 issue 或长任务基准；仓库内题目彼此相关，后续须同时报告逐题与仓库级配对。来源隔离不能证明底座预训练去污染。本次准入也不是模型能力验收，不允许用于参数发布。
