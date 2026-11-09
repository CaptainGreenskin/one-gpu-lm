# 014 — 训练覆盖与新仓库编码能力对照

状态：控制检查准备中，尚未开展模型评分。013 验证了训练状态和恢复；014 检查一个固定的 27B LoRA 候选是否提高任务成功率。

012 的前 96 条截取覆盖了 12/21 个训练任务，7 条失败任务的参考修复示范未被选中。本轮使用同一份已核验来源的 145 条原始记录，先过滤重复、超过 4096 上下文或 512 输出预算的记录，再按任务均衡分配 96 个名额，在各轨迹中均匀抽取位置。固定种子 101、96 次更新、rank-8 LoRA、学习率 2e-6；不追加教师 API。训练数据和推理评测数据仍隔离。

新的留出集来自固定版本的 JMESPath Python 实现和 fast-json-stable-stringify，计划各 6 个任务。所有任务必须通过干净源码、注入缺陷、恢复源码三组控制。原有 15 个 dev/test 任务改作历史回归，另外保留 9 个函数回归。这里测的是人为构造缺陷，不是历史 issue 修复成绩，也不是 SWE-bench。

在任何模型评分前冻结 [protocol.json](protocol.json) 的任务、数据哈希与门槛：新留出集净增至少 2 个成功任务，不能出现新留出或历史任务退步，9 个函数回归全通过，完整运行完成。通过仅表示可以进入多种子复验。单个种子、两个小仓库无法证明通用编码或长任务能力；底座预训练是否见过上游代码未知。

## 可复用入口

通用实现位于 `one_gpu_lm`，本目录保留配方、来源、冻结协议和报告。先按 [27B 运行手册](../../docs/27b-training.md) 安装隔离环境；私有路径只通过参数或环境变量传入。

```bash
python -m one_gpu_lm.data.balanced \
  --data "$ORIGINAL_DATA" --data-receipt "$ORIGINAL_RECEIPT" \
  --admission "$TRAIN_ADMISSION" --licenses "$TRAIN_SOURCE_LEDGER" \
  --model "$ONE_GPU_MODEL" --profile experiments/014-coding-quality/model-profile.json \
  --budget 96 --out runs/prepared-data

python -m one_gpu_lm.evaluation.install_node_deps \
  --node-root "$NODE_ROOT" --manifest experiments/014-coding-quality/node-dependencies.json \
  --out node-deps

python -m one_gpu_lm.evaluation.preflight \
  --root . --tasks private/tasks.json --out runs/admission
```

`prepared-data/train.jsonl` 和任务定义含私有内容；只公开 `selection.json` 中的覆盖计数、ID、长度与哈希。原始数据的来源、许可证、训练分区和入场记录由共享合同验证，缺少任意训练任务即拒绝准备。评测入场记录与训练入场记录分开保存，不能互换。

训练使用 `python -m one_gpu_lm train --config experiments/014-coding-quality/train-config.json --out runs/candidate`，配置的五个 `ONE_GPU_*` 输入同运行手册。必须先完成控制检查并冻结协议，再开展底座/候选配对评测。

本轮同时改变了样本筛选与 013 确定性训练运行时，因此它是候选与底座的质量筛查，不能把结果全部归因于均衡采样。源码、修复答案、轨迹、权重和优化器状态不公开；MIT 仅覆盖本项目原创代码，上游权利记录见 [sources.json](sources.json)。
