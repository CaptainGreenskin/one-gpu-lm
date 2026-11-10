# 单仓库内的训练系统模块

`one_gpu_lm` 是当前仓库编码训练链路的实现位置。保留一个主仓库，让数据契约、模型适配、训练、评测和服务的变更能在同一提交中接受验证。模块分开不等于已经形成六个成熟的独立产品。

| 模块 | 负责什么 | 主要实现 |
| --- | --- | --- |
| `common/` | 内容哈希、JSON 回执、动作格式、上下文策略、执行身份、检查点和硬件记录 | `agent_policy.py`、`receipts.py`、`training_data.py`、`checkpoints.py`、`runstate.py`、`telemetry.py` |
| `data/` | 私有训练记录、来源信息、训练分区约束、已验证轨迹/参考修复的准备 | `repository.py`、`balanced.py`、`function_records.py` |
| `models/` | 按 profile 验证/加载固定权重、模板、精度和本地生成 | `verify.py`、`hf_runtime.py` |
| `training/` | 配置预检、assistant-only LoRA、完整状态恢复、公开证据收集 | `config.py`、`doctor.py`、`runner.py`、`verify_resume.py`、`collect.py` |
| `evaluation/` | 任务准入、工具预算、离线工作区、独立评分、恢复、配对报告 | `preflight.py`、`agent.py`、`campaign.py`、`grading.py`、`report.py` |
| `serving/` | 模型进程与评测进程之间的本地请求协议和 Unix socket 服务 | `client.py`、`local_backend.py` |

模型适配层不启动服务，也不依赖评测或训练。训练直接调用模型适配层，不通过推理服务绕一圈。评测调用服务客户端，不加载 GPU 权重。数据准备可以调用独立评分器验证参考修复，但不启动 agent 或训练。共享动作契约供数据准备和 agent 同时使用，避免两份解析器各自变化。

当前 `serving/` 是实验用的本地串行后端，不是生产 HTTP 服务，也没有连续批处理或多租户保障。现有 `models/` 只是经过当前固定模型验证的 Hugging Face 适配器；新架构仍需加载、模板、缓存和梯度验收。导入这些模块不会加载 torch、模型权重或启动 GPU 工作。

## 从哪里运行

从仓库根目录、使用相应的 Python 环境运行。27B 的独立环境安装、训练和恢复见 [操作手册](../docs/27b-training.md)；仓库任务的隔离依赖另见 [仓库任务操作说明](../repo_lab/README.md)。

`python -m one_gpu_lm` 提供 `doctor`、`train`、`status`、`collect`、`verify-resume`、`evaluate` 和 `serve` 子命令。`train` 是新建的可恢复运行入口，旧 `repository_lora` 单次实验入口保留原有行为。

| 操作 | 新入口 | 兼容入口 |
| --- | --- | --- |
| 任务控制/准入 | `python -m one_gpu_lm.evaluation.preflight` | `python -m repo_lab.preflight` |
| 审计并均衡选择训练记录 | `python -m one_gpu_lm.data.balanced` | — |
| 构造训练记录 | `python -m one_gpu_lm.data.repository` | `python -m repo_lab.dataset` |
| 一轮 LoRA | `python -m one_gpu_lm.training.repository_lora` | `python -m repo_lab.train` |
| 启动本地模型服务 | `python -m one_gpu_lm.serving.local_backend` | `python -m repo_lab.backend` |
| 运行仓库任务 | `python -m one_gpu_lm.evaluation.campaign` | `python -m repo_lab.campaign` |
| 旧题回归 | `python -m one_gpu_lm.evaluation.regression` | `python -m repo_lab.regression` |
| 配对报告与曲线 | `python -m one_gpu_lm.evaluation.report` | `python -m repo_lab.report` |

参数保持兼容，可用 `--help` 查看。`repo_lab/` 的 Python 入口转发到同一份实现；`training_lab/` 保留 011 的历史 pilot，公共记录和隔离组件转发到新模块。新的通用能力写入 `one_gpu_lm/`，不在旧入口复制实现。

## 工具格式与上下文策略

[015 四组对照](../experiments/015-agent-interface/)通过冻结协议的 `variants` 显式选择策略，`campaign --variant NAME` 和 `agent --variant NAME` 使用同一入口。省略该参数时保留原提示和完整历史。

```json
{"variants": {"combined": {"format": "native_schema", "context": "bounded_history_v1"}}}
```

`native_schema` 为当前固定 Qwen 模板传入 `bash(command)` 工具定义，并使用相应提示；解析器和隔离执行器不变。`bounded_history_v1` 根据真实模板 token 数移除较早的完整交互对，保留任务和近期记录。它只裁剪发往模型的请求，完整审计轨迹和预算仍保留。每次请求记录裁剪前后 token 数、移除轮数、格式检查和耗时，失败的上下文请求也单独留证。

这套格式说明包含当前 Qwen 的 XML 约定。换模型时必须重新检查模板和解析兼容性；上下文裁剪也必须通过新任务对照，不能只凭“没有超限”推断效果改善。

## 实验、代码和模型版本

`experiments/` 保存实验协议、模型 profile、来源和结果；001–010 的从零训练/性能脚本保持各自的实验边界，不在这次迁移中强行改写。新的可复用实现放在模块里，实验目录描述怎样调用它。

012 的正式比较已在迁移前封存于提交 `fa7c5346d4af47101db5299f08df60eda1d84918`。迁移后的执行代码身份不同：新 campaign 会记录实际实现、共享代码、包初始化文件和 JS 运行时资产的哈希；旧 campaign 不能跨这次迁移直接 `--resume`。需要复现旧实现时检出记录中的原提交；旧命令可调用新实现不等于实验代码字节未变。已经发布的结果与哈希不重写。

代码、协议、指标和失败分析在 **one-gpu-lm**；原理知识在独立的 **infra-guide**；以后通过验收、满足分发条件的具体模型版本再交付 **ModelScope**。权重、原始数据和服务凭据不进入这个代码仓库。

## 什么时候拆仓库

评测工具出现独立使用者、独立依赖和发布周期，或推理服务需要独立部署与维护时，再将对应模块拆成仓库，并定义版本化的输入输出契约。当前仍以一个提交核对“数据 → 训练 → 评测”的一致性为主，不按目录数量衡量项目进展。

```bash
python -m unittest training_lab.test_contracts repo_lab.test_contracts tests.test_modules tests.test_training_infrastructure
# Linux、bwrap 和固定 agent 环境：
python -m unittest training_lab.test_sandbox_linux repo_lab.test_recovery_linux
```

模块边界和兼容 CLI 进入 CPU CI；真实 GPU 和 Linux 检查另留回执。结构调整本身不提升模型分数，本轮迁移也没有改变 012 未通过晋升门槛的结论。

本次已完成的 [迁移验收](../docs/validation/2026-11-09-modules/)包含 31 项 Linux/CPU 检查、六个仓库的原始版本测试、报告重建，以及真实 GPU 的生成一致性探针。
