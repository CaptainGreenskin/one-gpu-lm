# 27B 单卡训练操作手册

这套入口把环境检查、有限步数 LoRA、状态保存、换进程续训、回归评测和公开记录收集接起来。当前配方是固定版本的 Qwen3.8-27B、单张 RTX PRO 6000 Blackwell 96 GB、BF16 底座和 FP32 rank-8 LoRA。工程验收见 [013](../experiments/013-training-infrastructure/)；恢复通过不代表编码能力提升。

## 1. 准备独立环境

在 Linux x86_64、Python 3.10、支持 CUDA 13.0 的驱动上，从仓库根目录执行：

```bash
PYTHON_BIN=python3.10 bash scripts/setup_27b.sh .venv-27b
source .venv-27b/bin/activate
python -m one_gpu_lm --help
```

安装脚本创建不继承系统包的 venv，固定核心版本，从官方 PyTorch/PyPI 获取包，执行 `pip check`，保存 `installation.json` 和包含间接依赖的 `environment.lock.txt`。下载错误最多重试三次，保留 TLS 验证。已存在的环境不会被覆盖；失败环境先保留，再选择新目录重试。驱动不由脚本安装。实测的 61 包版本快照在 [constraints](../environments/pro6000-cu130-constraints.txt)，安装脚本使用它限制间接依赖版本；核心要求在 [environments](../environments/)。它固定包版本，不是 wheel 内容哈希锁。

GPU 必须空闲，配方要求至少 92 GiB 可用显存；预检还按检查点数量估算磁盘余量。它使用同一用户下的协作锁，不能阻止其他程序或其他用户抢占 GPU。单个例子的实际 token 长度决定显存开销，短样本验收并不保证所有 4096-token 数据都能装下。

## 2. 准备权重和合法训练记录

按 [011 的固定权重清单](../experiments/011-coding-model-selection/weights-manifest.json) 获取该 revision 的完整模型文件，保留其下载回执。加载器的原有下载检查和新的逐文件内容校验都要通过；已有缓存可复用。不要在恢复中途换基模、分词器、模板或精度。

训练输入使用 [仓库任务流程](../repo_lab/README.md) 产生的 `train.jsonl`、数据 receipt、任务 admission 和来源许可清单。该入口要求已准入的训练分区、固定仓库 revision，以及经过验证的轨迹或参考修复；不能把任意聊天 JSONL 改名后直接输入。更换数据时先完成准入和数据准备，再创建新 run。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export ONE_GPU_MODEL=/absolute/private/model-cache
export ONE_GPU_TRAIN_DATA=/absolute/private/dataset/train.jsonl
export ONE_GPU_DATA_RECEIPT=/absolute/private/dataset/receipt.json
export ONE_GPU_ADMISSION=/absolute/private/admission/admission.json
export ONE_GPU_LICENSES=/absolute/private/sources.json
```

这些路径示例需要替换成自己的本地文件。准入记录检查来源、分区和内容哈希，不替代对授权范围的判断。原始数据、参考答案、轨迹、主机信息和权重留在私有存储。

## 3. 检查并启动有限训练

```bash
python -m one_gpu_lm doctor --config configs/pro6000-27b.json \
  --out runs/preflight-001.json
python -m one_gpu_lm train --config configs/pro6000-27b.json \
  --out runs/coding-001
```

`doctor` 重新计算权重分片与元数据的内容哈希，检查数据身份、核心包、CUDA/BF16、显卡占用和磁盘。训练进程会再次预检，因此模型较大时启动需要时间。配置中的路径相对配置文件解析；环境变量填入的绝对路径只写入私有运行文件。

默认配方至多选择 96 条记录、更新 96 步，每 16 步提交检查点；每次进程的协作时间预算为 3600 秒。microbatch 为 1，一个例子对应一次优化器更新，仅走一遍选定顺序；所需步数超过可用样本数会拒绝。当前没有梯度累积、学习率调度、多轮 shuffle 或多卡训练。预算在完整更新边界检查，模型加载或单次 CUDA 操作不受硬超时强行打断。

调整配方请复制 JSON 并使用新 run。恢复要求原始配置、模型 profile、数据、实际环境和实现代码哈希一致，不允许在旧 run 上悄悄增加预算。

## 4. 查看状态、暂停和恢复

```bash
python -m one_gpu_lm status --run runs/coding-001
# 向自己启动的训练进程发送 SIGTERM；等待其在完整更新边界保存并退出。
# kill -TERM <owned_training_pid>
python -m one_gpu_lm train --config configs/pro6000-27b.json \
  --out runs/coding-001 --resume
```

正常完成退出码为 0，协作暂停为 75。异常退出先查本地私有日志，修复外部故障后才能恢复；代码或数据身份改变则必须新建 run。已完成 run 不可再次训练。

每个检查点包含 adapter、AdamW 状态、Python/torch CPU/CUDA 随机状态、样本顺序指纹、已提交指标与完整性清单。写入临时目录、同步并重命名后才更新 `latest.json`；失败的半写入不替换上一个指针。恢复先验证文件哈希，再恢复状态并核对指纹。未提交的指标保留到私有 orphan 文件，公开历史回退到已提交前缀。所有已提交检查点保留，需要按磁盘预算管理。

进程被强杀后只能从最后一个完整检查点继续，之间的更新可能重做。013 实测的是在第 4 步完整提交后退出，再由新进程恢复；它不证明任意 CUDA 内核中断、网络文件系统或整机断电均可无损恢复。检查点机制参考 [PyTorch 的完整训练状态保存](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) 和 [PEFT adapter 格式](https://huggingface.co/docs/peft/developer_guides/checkpoint)。

复跑恢复对照：

```bash
python -m one_gpu_lm verify-resume \
  --config experiments/013-training-infrastructure/recovery-config.json \
  --out runs/recovery-001 --interrupt-after 4
```

它按顺序运行连续 12 步、另起进程 4 步、换进程续到 12 步，核对 adapter、优化器、随机状态、逐步 loss、样本顺序和 16-token 贪心探针。数值容差在 [protocol.json](../experiments/013-training-infrastructure/protocol.json) 中提前固定；不能在看到失败结果后扩大容差来宣称通过。

## 5. 评测已保存的产物

```bash
python -m one_gpu_lm evaluate --run runs/coding-001 \
  --tasks /absolute/private/function-test.jsonl \
  --node /absolute/private/node/bin/node \
  --variant base --out runs/regression-base-001
python -m one_gpu_lm evaluate --run runs/coding-001 \
  --tasks /absolute/private/function-test.jsonl \
  --node /absolute/private/node/bin/node \
  --variant adapted --out runs/regression-adapted-001
```

这是使用既有函数题格式的工程回归入口，需要 Linux 用户命名空间、bwrap 和 Node。它校验检查点，启动自己拥有的本地 Unix socket 后端，运行隔离评分，退出时回收该后端。生成文本与后端日志留在 `private/`。更广泛的仓库任务、工具预算和独立留出仍走 [仓库评测](../repo_lab/README.md)。已有九题回归用于检测破坏，不能用于宣称达到先进编码模型水平。

需要手动实验服务时，`python -m one_gpu_lm serve --help` 提供原有后端入口；目前是串行本地服务，没有对公网部署或生产吞吐保障。权重和 adapter 不会因训练成功而自动发布。

## 6. 收集可公开证据

```bash
python -m one_gpu_lm collect --run runs/coding-001 \
  --out public/coding-001
```

收集器只复制白名单中的配置身份、指标、事件、逐进程硬件采样、加载与预检回执，并生成文件哈希。它不复制模型、adapter、优化器、原始数据和私有路径文件；遇到明显私密字段会拒绝。上传前仍需审查白名单中的内容，切勿提交整个 `runs/`。评测结果另审查 `artifact.json` 和 `metrics/result.json`，不复制其私有目录。

硬件采样每 5 秒一次，覆盖加载、更新、保存和探针；报告各进程的实际首末采样、最大间隔和采样错误。不同进程之间的停机不能算成同一段连续采样。训练更新时间与含校验/加载/检查点的总墙钟时间分别报告。

v1 的边界是单机单卡、固定版本的 27B LoRA。全参数训练、量化训练、多卡、调度平台和生产推理需要独立实现与实测。后续换基模沿用这些入口和 [升级协议](model-upgrades.md)，但仍须重新验证架构、梯度、模板、上下文和质量。
