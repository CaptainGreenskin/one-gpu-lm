# 021 配对行为观察

以下只描述已经执行并独立评分的配对结果，不重新运行任务，也不把相关性当作退步的唯一成因。首次编辑来自每次工具执行后的实际源码哈希；曾编辑与最终留下修改分开记录。

| 开发对照 | 通过 | 未编辑失败 | 编辑后失败 | 无效动作 / 调用 |
| --- | --- | --- | --- | --- |
| dev-base | 6/15 | 6 | 3 | 4/152 |
| dev-reference020 | 8/15 | 3 | 4 | 0/147 |
| dev-observed | 6/15 | 7 | 2 | 6/149 |

| 对照 | 任务 | 新候选首次编辑调用 | 最终修改文件数 | 调用数 | 退出状态 | 判分原因 |
| --- | --- | --- | --- | --- | --- | --- |
| dev-vs-base | fast-deep-equal-regexp-flags | 未编辑 | 0 | 12 | LimitsExceeded | runtime_error |
| dev-vs-reference020 | fast-deep-equal-map-values | 未编辑 | 0 | 12 | LimitsExceeded | runtime_error |
| dev-vs-reference020 | fast-deep-equal-regexp-flags | 未编辑 | 0 | 12 | LimitsExceeded | runtime_error |

达到单次生成长度上限且动作解析失败的记录：more-itertools-windowed（3 次）、p-limit-pending-count（3 次）。这是从已保存输出和解析结果核对出的共现，未增加生成预算验证因果；不能据此推断放宽预算就会修复成功。

新增退步共 2 道不同任务，均未发生源码编辑；同一任务可能在两组对照中重复列出。上述格式失败与这些退步任务应分开分析，不能合并成同一个原因。

所有配对（包括改善、不变和退步）见 [paired-outcomes.csv](paired-outcomes.csv)。
本轮 84 条实际状态替换中，类别分布为 {'tool_error': 6, 'wrong_edit': 3, 'read_only': 75}。数据覆盖主要来自尚未编辑的历史；不能把这次小样本单种子结果外推为真实轨迹训练普遍无效。
训练损失来自不同训练样本，不能替代独立任务通过率；开发分区已重复使用，模型发布仍取决于事先固定的最终门槛。
