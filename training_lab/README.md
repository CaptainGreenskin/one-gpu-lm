# 可随底座迭代复用的训练与验收组件

目标：未来接入新的千问或其他基模时，沿用相同的数据准入、任务执行、训练记录、对照评测和产物格式，用证据判断新版本是否值得采用。当前实现是单卡、BF16、LoRA、函数级编码任务的首轮试验入口；不是已完成的通用训练平台。

## 已实现的边界

| 通用部分 | 每个新模型需要提供或重新验收的部分 |
| --- | --- |
| JSONL 数据契约、来源记录、内容哈希与精确去重 | 官方模型 ID、锁定提交、权重/配置/许可证哈希 |
| Python / JavaScript / C++ 的离线执行与逐用例判分 | Transformers 加载类、dtype、attention 后端 |
| 冻结底座的 LoRA 更新、梯度/参数变更检查、保存重载 | LoRA 目标层、chat template、特殊 token 和输出约定 |
| 原始模型/训练后相同任务、相同生成预算配对 | 新架构的加载完整性、数值、缓存、内核和显存验证 |
| 版本化 JSON 结果、环境/源码哈希、训练曲线与三份报告 | 新版本实测结果，不能复用旧模型成绩 |

`pilot.py` 不导入 Qwen 专属类。当前模型差异放在 `experiments/011-coding-model-selection/pilot-profile.json`；允许两种标准 AutoModel 加载入口。遇到新的架构时可以新增经过审查的加载适配，而不是在所有脚本中加入模型名称判断。关闭远程代码执行和隐式 CPU 卸载；缺少权重、加载不匹配、数据越过长度预算都会终止。

## 输入契约 v1

每行一个 JSON 对象，字段如下。原始文件由运行者提供并留在受控目录；仓库中不放本次训练语料。

| 字段 | 含义 |
| --- | --- |
| `id`, `language`, `prompt` | 唯一任务 ID；`python` / `javascript` / `cpp`；完整模型请求 |
| `target` / `reference` | train 的目标答案 / test 的参考实现，二者分开 |
| `cases` | `{input, expected}` 数组；所有用例通过才算任务通过 |
| `provenance.kind`, `.source` | 素材类型与来源/版本说明 |
| `provenance.rights_basis`, `.training_allowed` | 权利依据及明确的训练使用准入，缺失即拒绝 |
| `provenance.redistribution_allowed` | 另行记录可否公开原文，默认否 |

Python/JavaScript 实现 `solve(x)`，以 JSON 输入输出。C++ 实现 `long long solve(const std::vector<long long>&)`，当前仅支持整数数组到整数。参考实现必须先通过所有用例。相同 ID 或完全相同提示跨越训练/测试会被拒绝；语义相近、翻译、仓库和任务家族泄漏仍需准备阶段审查，不能把哈希去重称为无污染证明。来源字段是审计记录，不能代替法律判断。

## 运行

Linux 需要可用的 `bwrap` 用户命名空间、Python 3、g++，以及显式指定的 Node 可执行文件。执行器不挂载用户主目录，不开放网络，使用 CPU/虚拟内存/文件/输出/时间限制；Node 另限堆大小。这是受控试验隔离，尚不是运行恶意用户代码的多租户安全沙箱，也没有做完整 resident-memory/cgroup 限额。

```bash
python -m unittest training_lab.test_contracts
python -m training_lab.pilot \
  --profile experiments/011-coding-model-selection/pilot-profile.json \
  --model models/qwen27 --train data/train.jsonl --test data/test.jsonl \
  --node /absolute/path/to/node --out runs/reference-check --validate-only
python -m training_lab.pilot \
  --profile experiments/011-coding-model-selection/pilot-profile.json \
  --model models/qwen27 --train data/train.jsonl --test data/test.jsonl \
  --node /absolute/path/to/node --out runs/pilot-001
python -m training_lab.report --run runs/pilot-001
```

`--model` 需要带有固定版本下载校验回执 `download-receipt.json` 的本地模型。011 的 `download.py` 是首个模型来源实现，允许官方 Hugging Face / ModelScope 下载，但必须通过同一份锁定哈希。后续模型应生成独立 manifest；不能只改模型名称后沿用旧权重哈希。

输出目录不可覆盖。遇到失败保留原记录，以新 run ID 修复复测。完整输出包括私有内容；发布时只采用 `report.py` 生成的 `artifacts.json` 文件清单，**不要把 runs 目录整体上传**。适配器、生成代码和原始 JSONL 默认不公开。安装版本以本次结果中的 `environment.json` 为准；新架构升级依赖时创建隔离环境并重新验收。

## 有意义的积累与下一步

GitHub 保存实现、配置、来源清单、协议、公开指标、曲线、问题复盘和复现入口。ModelScope 承载通过验收的具体模型版本及其模型卡、加载方法和关联结果；本阶段只生成模型卡草稿，不上传权重。infra-guide 记录解释与操作知识，并链接回实验事实。三者共同使用版本号和内容哈希建立关联，避免文档、代码和权重各说各话。

每个候选模型都应先跑冻结的评测协议，再决定是否训练；保留与旧版本相同任务、相同 agent 工具、相同 token/时间预算的对照。随着当前小样本被开发人员反复查看，它只能作为回归集，不能继续充当最终秘密测试集。

尚未完成：真实仓库级隔离评测、稳定留出/盲测、长任务与中断恢复、端到端部署服务、TTFT/解码/缓存性能对照、多种架构验证、蒸馏数据入口。不能把这次的函数级 smoke 得分称为这些能力。下一阶段优先补真实仓库任务和对照，再扩大数据与卡时。
