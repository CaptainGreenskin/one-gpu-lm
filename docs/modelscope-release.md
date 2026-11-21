# 27B 编码模型发布目标

首个发布物是通过验收的 27B 编码 LoRA，发布到 ModelScope 国际站的 CaptainGreenskin 账号。认证、发布与公开下载均使用该账号所在站点。代码、评测协议、匿名曲线和复现入口保存在 `one-gpu-lm`。模型卡固定底座及 adapter 的版本、许可、依赖和加载命令。合并权重另经导出与加载验证。[ModelScope 发布账号](https://www.modelscope.ai/profile/CaptainGreenskin) · [官方站点说明](https://www.modelscope.cn/skills/modelscope/ms-hub)

| 阶段 | 验收证据 |
|---|---|
| 验证推理流程 | [038](../experiments/038-independent-compact/) 在四个新家族的 24 道普通修复题上，等预算比较旧前缀和固定失败摘录。报告净增、退步和卡时，按冻结门槛选择下轮的流程。 |
| 验证参数收益 | 同一底座、推理流程和预算下配对比较基础权重与候选 LoRA。按[既有独立开发目标](../experiments/025-verified-distillation/cycle-spec.json)冻结至少 60 道开发题、六个家族、两种语言和三轮复跑。目标为至少 +10 个百分点、配对区间下界大于零、单个家族下降不超过 10 个百分点。 |
| 独立验收 | 开发达标后才启用封存题。沿用 [022 的完整发布门槛](../experiments/022-action-repair/protocol.json)，24 道新任务与 48 道历史回归完整配对，新任务净增至少三道，两部分零退步，9 道函数回归全通过。正式运行前核验新题的家族与任务从未进入训练或开发选择。 |
| 发布和复验 | 校验底座及训练来源的发布授权、代码和训练产物哈希、工具格式错误率、Greedy 生成保存重载一致性、敏感内容扫描和执行审计。上传脱敏模型卡、`safetensors` adapter、加载入口及联结评测的版本信息，从公开版本重新下载验哈希和加载。 |

038 的独立复验已结束。失败摘录净增两道，低于三道的门槛。下一轮参数对照统一沿用旧前缀。[038 结果与曲线](../experiments/038-independent-compact/results/2026-11-20/)

039 在六个新家族的 60 道题上与底座同预算对照。031 LoRA 与底座各完成 48/180，开发净变化 0 个百分点，未通过开发门槛。最终测试集仍封存。031 的训练输入、教师模型和上游许可材料已绑定到同一候选。[039 完整审计与曲线](../experiments/039-release-development/results/2026-11-20/) · [031 来源与许可核验](../experiments/031-correction-mix/release-provenance.md)

041 在九道训练题上采集到 27 条经审计的接续训练记录。042 将这些记录加入既有训练样本，完成 126 次更新并验证重载生成一致。独立开发对照使用六个新仓库家族的 60 道题，每题以同一推理流程和预算分别运行底座与 LoRA 各三次。完整审计后，底座完成 30/180，LoRA 完成 24/180。042 未通过开发门槛。[041 采集结果](../experiments/041-first-edit-continuation/results/2026-11-21/) · [042 训练曲线](../experiments/042-first-edit-sft/results/2026-11-21-training/) · [042 独立对照与硬件记录](../experiments/042-first-edit-sft/results/2026-11-21-development/)

最终验收从 [022 的公开任务清单](../experiments/022-action-repair/results/2026-11-12/task-inventory.json)绑定两组任务身份。新留出包含四个仓库家族，与 039 已核对的训练及开发来源无重叠。封存文件完整性以原协议哈希为准。[晋级检查](../experiments/039-release-development/promotion.py)重算 039 的开发门槛，并核对入选适配器和封存清单的身份。

[发布打包入口](../experiments/039-release-development/release_package.py)重算已经回放审计的最终配对分数，核对同一适配器的分发审查、许可原文和两次独立重载。全部门槛通过后，才在新目录生成 `payload/` 和 `release-evidence.json`。前者包含两份适配器文件、带聚合结果和加载示例的模型卡，以及核验过的许可材料。后者保留在本地，记录逐文件大小与 SHA-256。发布后用实际 ModelScope 提交号补全下载校验回执。

## 恢复适配器备份

已有 031 私有备份可用以下命令恢复。运行环境需安装 NumPy 与 safetensors。命令在 CPU 上核对 SHA-256、底座 revision、目标模块、LoRA 秩和全部张量，并复查恢复目录。发布版本下载后还需在匹配底座上执行固定输入的加载和生成复验。

```bash
python -m one_gpu_lm.models.adapter_bundle \
  --adapter /path/to/backup \
  --profile experiments/031-correction-mix/candidate-profile.json \
  --receipt experiments/031-correction-mix/results/2026-11-17-training/integrity.json \
  --restore-to /path/to/restored-adapter
```

042 已用私有备份在独立目录完成一次 CPU 恢复，核验逐文件哈希并读取全部 992 个张量。用上述命令恢复 042 时，`--profile` 和 `--receipt` 分别使用备份内的 `training-profile.json` 和 `integrity.json`。

## 绑定候选训练记录

后续实验的晋级入口可将通过开发复验的授权、同一候选的适配器和经核验的训练配方交给 `run_approved`，随后用 `audit_approved` 复核原登记的最终测试。打包时成组提供 `--training-protocol`、`--training-profile` 和 `--training-integrity`，核对晋级授权、训练数据、来源清单、配方及适配器哈希。模型卡根据通过核验的记录填写候选名称和训练种子。打包同时重算原最终验收门槛，并核对逐文件许可材料和独立重载回执。[最终验收入口](../experiments/039-release-development/final_campaign.py) · [打包入口](../experiments/039-release-development/release_package.py) · [训练来源校验](../one_gpu_lm/models/release_lineage.py)

041 的九份接续状态使用与 031 相同的五个训练来源和固定版本。已复核工作站快照、上游模型许可清单及十份许可与署名文件的原始字节。打包时按实际入选训练来源的版本复用材料，分发审查继续绑定具体候选、训练数据和拟发布文件。[041 来源回执](../experiments/041-first-edit-continuation/source-notices.json) · [031 来源说明](../experiments/031-correction-mix/release-provenance.md) · [Apache-2.0 分发条款](https://www.apache.org/licenses/LICENSE-2.0#redistribution)

## 核对 ModelScope 上的发布版本

[发布入口](../experiments/039-release-development/publish_package.py)使用通过验收的打包目录和 ModelScope 国际站的登录态。在本机安装 `modelscope-hub==0.7.0` 后，`check` 核对回执与本地文件。`stage` 创建私有模型仓库，只上传经审核的文件，从实际 Git 提交重新下载并逐文件校验。`publish` 再次比对同一提交后开放访问，匿名下载到新目录。状态保存在打包目录的 `modelscope-release.json`，网络中断后可按原提交恢复。[ModelScope Hub SDK](https://github.com/modelscope/modelscope_hub)

```bash
PYTHONPATH=. python experiments/039-release-development/publish_package.py check --package /path/to/accepted-release
PYTHONPATH=. python experiments/039-release-development/publish_package.py stage --package /path/to/accepted-release
PYTHONPATH=. python experiments/039-release-development/publish_package.py publish \
  --package /path/to/accepted-release --download-to /path/to/fresh-download
```

`modelscope-release.json` 中的 `staged_manifest` 包含站点 `endpoint`、仓库 `repo_id`、完整 ModelScope Git 提交 `revision`，以及逐文件的 `files` 清单，每项有 `bytes` 和 `sha256`。可以将该字段单独保存为 JSON 后独立复验。回读先匿名核对仓库、公开状态和完整文件清单，再将指定提交逐文件下载到新目录并计算哈希。[下载校验实现](../one_gpu_lm/models/modelscope_readback.py)

```bash
python -m one_gpu_lm.models.modelscope_readback \
  --manifest /path/to/release-files.json \
  --out /path/to/fresh-download
```

发布前，在空闲 GPU 上分别独立加载原始适配器和恢复到新目录的副本。两次使用同一份底座文件清单、评测配置和适配器哈希回执。回执记录两个固定输入的 token 哈希与 greedy 输出 token 哈希。

```bash
python -m one_gpu_lm.verified.reload_probe capture \
  --model /path/to/verified-base --adapter /path/to/adapter \
  --profile /path/to/evaluation-profile.json \
  --base-manifest /path/to/base-files.json \
  --adapter-receipt /path/to/approved-adapter-hashes.json \
  --out /path/to/before.json

python -m one_gpu_lm.verified.reload_probe compare \
  --before /path/to/before.json --after /path/to/after.json \
  --out /path/to/reload-comparison.json
```

第二次 `capture` 使用恢复后的适配器目录，将输出写入 `after.json`。候选的 LoRA 配方与评测配置里的默认配方有差异时，两次都通过 `--adapter-profile` 提供经审计的配方。公开下载校验完成后，将下载目录再做一次 `capture`，与发布前的 `before.json` 比较。[重载校验实现](../one_gpu_lm/verified/reload_probe.py)

031、032 等历史候选保留已验证备份，模型卡会注明测试任务分布、训练种子和局限。短仓库修复结果描述本轮预算内的能力。长任务另有独立证据后在发布记录中报告。若候选落败，公开该轮开发结论，后续候选继续使用封存边界。
