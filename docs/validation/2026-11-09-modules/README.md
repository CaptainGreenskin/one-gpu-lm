# 模块迁移验收

迁移前结果提交：`fa7c5346d4af47101db5299f08df60eda1d84918`。本次只调整当前仓库编码链路的实现边界，不改变 012 的训练配置、任务预算、既有结果或晋升判定。

| 检查 | 实际结果 |
| --- | --- |
| CPU 数据、评分、兼容入口和模块边界 | 24 项通过；两套 CLI 的 `--help` 可在没有 GPU 库的环境中运行 |
| 固定 Linux 环境的完整检查 | 31 项通过，包含上述 CPU 检查、bwrap 隔离与真实控制进程恢复；见 `linux-tests.txt` |
| 六个真实仓库的原始版本测试 | 6/6 通过，验证迁移后的 Python/Node 依赖挂载与 JS 测试资产；见 `clean-controls.json` |
| 实现迁移对照 | 20 个函数/类的 AST 与迁移前相同，两个 JS 运行时资产字节未变；见 `local-checks.json` |
| 已封存报告重建 | 新旧入口均重现相同的 `summary.json`、`hardware.json`、`comparison.png`、`loss.png` |
| RTX PRO 6000 上的真实模型检查 | 固定底座及既有适配器重新加载，16 个贪心 token 与迁移前保存重载探针逐个一致；见 `gpu.json` |

GPU 加载没有 missing/unexpected/mismatched keys 或加载错误，回执见 `loading.json`。GPU 探针使用原训练上下文，仅检查加载和生成的一致性；不是新的质量分数，也没有再次训练或重跑完整 36 题。探针完成后进程退出，GPU 回到空闲。

AST 对照采用 `ast.dump(include_attributes=False)`，只覆盖清单中的实现；导入关系、兼容转发、报告 CLI 包装和执行源码身份记录有意改变，另由命令行、Linux 与 GPU 检查覆盖。过去的源码身份仍对应过去的结果，不能把这次结构变化描述成字节级无修改。

`artifacts.json` 记录公开检查文件及被检查源码的哈希。原始任务、生成内容、权重和主机信息未包含在这些回执中。整体模块职责与兼容入口见 [one_gpu_lm](../../../one_gpu_lm/README.md)。
