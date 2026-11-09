# 仓库任务 → 本地 agent → 一次后训练 → 配对验收

实现已迁至 [one_gpu_lm 的分层模块](../one_gpu_lm/README.md)。本目录保留兼容入口和原有操作说明；下列 `repo_lab` 命令仍调用同一份新实现。迁移前的实验身份与结果保留，不能跨源码身份变化恢复旧 campaign。

这一层复用 mini-swe-agent 的控制循环，增加本地模型后端、离线源码工作区、独立判分和完成步骤后的恢复。实验 012 使用真实开源仓库中的构造缺陷；它不是 SWE-bench 历史 issue 成绩，也尚不是日用模型认证。

## 可复用边界

| 部分 | 更换底座时如何处理 |
| --- | --- |
| `backend.py` + 模型 profile | 更换加载类、模板参数与模型身份；新架构重新验证加载、生成、缓存和 LoRA 目标层 |
| `agent.py` | 固定提示、工具接口与预算；当前支持唯一的 Bash 围栏动作或 Qwen XML Bash 动作 |
| `environment.py` / `grading.py` | 沿用离线工作区和独立判分；增加语言时补充只读运行时与参考控制 |
| `preflight.py` | 每个新任务检查原始通过、缺陷失败、恢复后通过；检测不到缺陷的任务不能计分 |
| `dataset.py` / `train.py` | 仅从训练分区收集通过验证的轨迹或参考修复；前缀不参与 loss，按固定顺序和预算选择样本 |
| `campaign.py` / `regression.py` | 保留任务身份与已消费预算，做同题配对和既有能力回归 |

当前实现要求 Linux 用户命名空间和 bwrap，agent 环境使用 Python 3.10，模型运行在独立 Python 环境；Node 使用显式固定版本路径。硬件和依赖记录见 [012 运行时记录](../experiments/012-repository-agent/preflight/runtime-lock.json)。CPU 测试通过不代表另一台机器的隔离或 GPU 已可用。

## 换成自己的合法任务

准备一个私有根目录，放置 `sources/<owner>--<repo>/` 的固定源码快照、`.venv-agent/`、`node-deps/<owner>--<repo>/node_modules/`（按需）及 `runtime.json`。`runtime.json` 的 `node_root` 指向完整 Node SDK；`model`、`model_python` 只供你自己的启动脚本使用。源码目录保留版权和许可文件，不提供 `.git` 历史、未修改的构建产物或其他答案副本。

任务 JSON 的顶层是 `{"schema_version": 1, "tasks": [...]}`，每条任务包含：

- `id`、`repository`、固定 `revision`、`split`（train/dev/test）和 `family`。
- `description`、明确的 `editable_files`、在工作区内执行的 `regression_command`。
- `mutations`：`python_stub` 指定 `path` / `symbol`；或 `replace` 指定 `path` / 唯一 `before` / `after`。
- 来源、许可及拆分的单独台账。任务和命令由评测集维护者审查；不能将任意网络任务自动视为可信配置。

按仓库和任务家族拆分；训练集不得包含开发/测试的答案或轨迹。真实基模可能在预训练中见过公开仓库，因此“本轮未训练过”不等于“基模从未见过”。评测测试可供 agent 阅读，终局检查使用独立原始副本；这里没有声称存在另一套秘密测试断言。

先运行控制，再将任务哈希、准入回执哈希、实际数量和预算写入独立 protocol。`campaign.py` 要求 `status=frozen_before_model_rollouts` 且 `admission_sha256` 匹配；该字段必须真实反映你的运行顺序，不能事后补写成预注册。实验 012 的接口修订单独披露。

```bash
python -m repo_lab.preflight --root RUN_ROOT --tasks PRIVATE_TASKS.json \
  --out CONTROL_OUT

# 使用模型 Python 环境；套接字留在工具工作区之外。
python -m repo_lab.backend --model LOCAL_VERIFIED_MODEL --profile MODEL_PROFILE.json \
  --socket PRIVATE_SOCKET --out BACKEND_RECEIPTS

# 使用固定的 agent Python 环境；每个 task 由独立控制进程运行。
python -m repo_lab.campaign --root RUN_ROOT --tasks PRIVATE_TASKS.json \
  --admission CONTROL_OUT/admission.json --protocol FROZEN_PROTOCOL.json \
  --socket PRIVATE_SOCKET --out BASELINE_OUT

python -m repo_lab.dataset --root RUN_ROOT --tasks PRIVATE_TASKS.json \
  --campaign BASELINE_OUT --out PRIVATE_DATASET_OUT

# 先正常停止推理后端，释放显存；仅运行这一轮预定训练。
python -m repo_lab.train --model LOCAL_VERIFIED_MODEL --profile MODEL_PROFILE.json \
  --protocol FROZEN_PROTOCOL.json --admission CONTROL_OUT/admission.json \
  --data PRIVATE_DATASET_OUT/train.jsonl --out CANDIDATE_OUT
```

随后用 `backend.py --adapter CANDIDATE_OUT/private/adapter` 启动候选，使用同一个 protocol 和任务清单运行第二次 campaign。不得把模型“测试通过”的文字当成绩：`grading.py` 仅复制允许的源码改动到新副本，重跑声明的测试，检查原有受保护文件是否被篡改。达到调用上限时源码仍可被判分，退出原因与任务是否通过分别记录。

## 恢复与验证

`--recovery-task TASK_ID` 在该任务第二个完整步骤后结束控制进程，再由新进程读取检查点恢复。保留消息、工作区哈希、模型身份、已用调用/token/时间预算；`in_flight` 状态拒绝自动重放，避免重复执行有副作用的命令。不是任意命令执行中断都能恢复。

```bash
python -m unittest training_lab.test_contracts repo_lab.test_contracts
# 在固定 Linux agent 环境中，而且模型已经加载后，再检查共存情况：
python -m unittest training_lab.test_sandbox_linux repo_lab.test_recovery_linux
```

工具工作区无外部网络，主目录和模型目录未挂载，依赖只读。`RLIMIT_NPROC` 是共享用户的线程/进程保护上限，不能当作单任务 cgroup 配额。本实现面向专用实验机，不是多租户不可信代码托管服务。

公开协议、来源身份、准入/失败回执、代码、曲线与配对指标。源码快照、原始任务、轨迹、补丁和权重保留在私有运行目录；不得整体上传 `runs/`。更换底座时重跑相同任务，同时补充未被反复调参的新留出集。只有通过事先写明的门槛，才讨论晋升候选；小样本成功不直接等于生产或日用验收。
