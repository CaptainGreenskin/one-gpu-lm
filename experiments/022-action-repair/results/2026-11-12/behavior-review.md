# 022 配对行为观察

以下只描述已经执行并独立评分的配对结果，不重新运行任务，也不把相关性当作退步的唯一成因。首次编辑来自每次工具执行后的实际源码哈希；曾编辑与最终留下修改分开记录。

| 开发对照 | 通过 | 未编辑失败 | 编辑后失败 | 无效动作 / 调用 |
| --- | --- | --- | --- | --- |
| dev-base | 7/15 | 6 | 2 | 4/149 |
| dev-reference020 | 6/15 | 4 | 5 | 0/150 |
| dev-action | 7/15 | 4 | 4 | 1/149 |

| 对照 | 任务 | 新候选首次编辑调用 | 最终修改文件数 | 调用数 | 退出状态 | 判分原因 |
| --- | --- | --- | --- | --- | --- | --- |
| dev-vs-base | fast-deep-equal-regexp-flags | 未编辑 | 0 | 12 | LimitsExceeded | runtime_error |
| dev-vs-reference020 | fast-deep-equal-array-members | 未编辑 | 0 | 12 | LimitsExceeded | runtime_error |

021 中两道退步任务在本轮同预算下的结果（已用于诊断的开发题，不是独立最终测试）：

| 任务 | 对照 | 通过 | 首次编辑调用 | 最终修改文件数 | 总调用 |
| --- | --- | --- | --- | --- | --- |
| fast-deep-equal-map-values | dev-base | False | 未编辑 | 0 | 12 |
| fast-deep-equal-map-values | dev-reference020 | False | 11 | 1 | 12 |
| fast-deep-equal-map-values | dev-action | False | 12 | 1 | 12 |
| fast-deep-equal-regexp-flags | dev-base | True | 5 | 1 | 7 |
| fast-deep-equal-regexp-flags | dev-reference020 | False | 未编辑 | 0 | 12 |
| fast-deep-equal-regexp-flags | dev-action | False | 未编辑 | 0 | 12 |

达到单次生成长度上限且动作解析失败的记录：p-limit-pending-count（1 次）。这是从已保存输出和解析结果核对出的共现，未增加生成预算验证因果；不能据此推断放宽预算就会修复成功。

新增退步共 2 道不同任务，均未发生源码编辑；同一任务可能在两组对照中重复列出。上述格式失败与这些退步任务应分开分析，不能合并成同一个原因。

固定对照相较 021 的重新运行（只描述波动，不参与选候选）：

| 固定对照 | 021 通过数 | 本轮通过数 | 本轮改善 | 本轮退步 |
| --- | --- | --- | --- | --- |
| dev-base | 6/15 | 7/15 | more-itertools-split_at | 无 |
| dev-reference020 | 8/15 | 6/15 | 无 | fast-deep-equal-map-values, fast-deep-equal-regexp-flags |

这些跨轮差异不归因给新训练。相同权重与调用上限不保证整条工具轨迹逐字一致；本轮选择只使用本轮完整三组，不能择优使用历史对照。

对固定 020 的两道变化题进一步只读检查：首个模型输入和输出均与 021 完全相同；首条工具反馈则不同，但只归一化其中的时间字段后，反馈全文相同。新旧轨迹各含 17 处匹配的日期/时间字段。后续实际模型输入已发生差异；这说明整条工具观察不是逐字固定的。
这个检查没有改写真实输入、没有补跑模型，也没有隔离时间字段对最终分数的因果影响。不能据此声称时间戳是退步的唯一原因。见 [输入反馈检查回执](rerun-variation-audit.json) 与 [只读检查实现](../../audit_reruns.py)。

所有配对（包括改善、不变和退步）见 [paired-outcomes.csv](paired-outcomes.csv)。
本轮 55 条提前行动替换中，两种训练情境分布为 {'evidence_then_edit': 28, 'missing_read_then_edit': 27}。保留了全部原有失败恢复样本（28 条锚点失败、27 条错误编辑）；这是组合课程，不能单独归因到某一项改动。
训练损失来自不同训练样本，不能替代独立任务通过率；开发分区已重复使用，模型发布仍取决于事先固定的最终门槛。
