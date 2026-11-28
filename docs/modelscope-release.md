# 27B 编码模型发布目标

**当前状态 · 058 在完整审计的独立开发对照中提高 11.67 个百分点，达到开发门槛。** 封存测试按既定协议验收。ModelScope 首发还需通过最终验收、许可与重载检查。 [058 开发曲线与结果](../experiments/058-read-history-replay/results/2026-11-28-development/) · [058 训练曲线](../experiments/058-read-history-replay/results/2026-11-28-training/)

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

044 加入符合正式评测流程的完整修复轨迹，完成 126 次更新。训练来源复核将 1,116 条动作、五个上游仓库的固定版本和十份许可与署名文件绑定到该适配器，底座许可为 Apache-2.0。六个新仓库家族的 60 道题各运行三轮配对，底座修对并提交 27/180，LoRA 21/180。开发审计完整，候选未达到净提升门槛，最终测试仍封存。 [044 训练记录](../experiments/044-deployment-recovery-sft/results/2026-11-22-training/) · [044 独立开发对照](../experiments/044-deployment-recovery-sft/results/2026-11-22-development/)

045 和 047 都从每个训练状态执行两条轨迹，分别使用贪心和采样首步，再用相同规则核对修复和提交。045 得到 5 对偏好样本，047 在 129 对训练状态上得到 16 对。048 用合计 21 对样本训练 Qwen3.8-27B 的秩 8 LoRA，完成 6 次更新。独立开发对照在六个新仓库家族的 60 道题上各做三轮配对。两组统一使用 038 保留的原始开场和相同预算，按登记的门槛从完整回放审计中判断是否晋级。 [045 配对结果](../experiments/045-first-action-forks/results/2026-11-22/) · [047 配对结果](../experiments/047-expanded-action-forks/results/2026-11-22/) · [048 训练曲线](../experiments/048-expanded-action-preference/results/2026-11-23-training/) · [048 开发登记](../experiments/048-expanded-action-preference/development-registration.json)

048 的 360 次开发评测已通过完整执行审计。底座修对并提交 24/180，LoRA 为 27/180，净增 1.67 个百分点，未达到开发门槛。两组始终未改码的 47 道题中，有 45 道在每轮都向两组模型提供过原始可编辑源码片段。最终测试继续封存。 [048 独立开发结果](../experiments/048-expanded-action-preference/results/2026-11-23-development/) · [未改码诊断](../experiments/048-expanded-action-preference/#开发结果)

049 在 13 个已查阅源码仍未改码的训练状态上，用相同的剩余预算比较原续跑和固定改码提示。提示组有两个修复通过独立评分，两组均未完成验证并提交。此次采集没有达到扩大门槛。 [049 配对结果和曲线](../experiments/049-guided-repair-suffix/results/2026-11-23/)

050 实际重放 59 条经审查的历史修复。39 条保留完整的改码、回归和提交示范，合计 160 个训练动作。5 条是必要多文件修复，低于 6 条门槛。20 条在动作规划时遇到单个补丁片段超过 512 token 上限。 [050 首轮结果和曲线](../experiments/050-verified-repair-closure/results/2026-11-23/)

050 补充轮按原预算重放这 20 条，新增 15 条合格示范。合并后有 54 条示范和 243 个训练动作，其中 8 条是必要多文件修复，达到数据准入门槛。 [050 补充轮结果和曲线](../experiments/050-verified-repair-closure/results/2026-11-23-supplement/)

051 用这 243 个动作从固定的 27B 底座训练秩 8 LoRA，完整执行 31 次更新，通过保存后重载和参数审计。独立开发对照已冻结 6 个新家族，每家族 10 道题，各做三轮配对。 [训练曲线与硬件测量](../experiments/051-repair-closure-sft/results/2026-11-23-training/) · [051 开发登记](../experiments/051-repair-closure-sft/development-registration.json)

051 开发对照完成 360 次评测，底座修对并提交 35／180，LoRA 为 44／180。净增 5.0 个百分点，配对区间下界为 −6.11 个百分点，单家族最大退步 13.33 个百分点。051 未达到开发门槛，最终测试仍封存。 [051 开发结果与曲线](../experiments/051-repair-closure-sft/results/2026-11-24-development/)

052 训练使用 67 条经重放验证的示范，要求模型在首次修改前能看到所需源码。独立开发对照完成 360 次评测，底座修对并提交 20／180，LoRA 为 27／180，净增 3.89 个百分点，配对区间下界为 −3.33 个百分点。实际改码次数增加 15 次，改码后未修对的次数增加 14 次。052 未达到开发门槛，最终测试继续封存。 [052 训练与开发记录](../experiments/052-grounded-first-edit/)

053 新增在部分补丁上实际执行的回归检查和看到失败后的修改示范，完成 41 次优化器更新。独立开发对照中，修对并提交的次数从 36／180 增至 54／180，净增 10 个百分点，配对 95% 区间为 −1.67 至 +23.33 个百分点。区间下界未达到晋级门槛。 [053 开发结果与曲线](../experiments/053-post-edit-recheck/results/2026-11-25-development/)

054 使用 79 个修复案例、394 条动作和 50 次优化器更新，适配器已通过独立恢复校验。新的 60 道开发题已完成离线参考修复核验，登记底座与 LoRA 的 360 次同预算对照。 [054 训练曲线与硬件记录](../experiments/054-expanded-repair-sft/results/2026-11-25-training/) · [054 开发验收登记](../experiments/054-expanded-repair-sft/development-registration.json)

054 的开发对照已完成 180 个配对。底座修对并提交 38／180，LoRA 为 77／180。配对提升 21.67 个百分点，95% 区间为 +6.67 至 +38.33 个百分点，满足全部开发门槛。11 道必要多文件题两组均未完成。 [054 开发结果、曲线与卡时](../experiments/054-expanded-repair-sft/results/2026-11-26-development/)

054 在 24 道新留出题上从 10／24 提高到 14／24，其中新增成功 6 道、退步 2 道。在 48 道历史回归题上从 25／48 提高到 30／48，其中新增成功 6 道、退步 1 道。工具调用格式及 9 道函数回归达到门槛。两组仓库题均出现退步，因而未通过最终质量门槛。[054 最终验收与调用曲线](../experiments/054-expanded-repair-sft/results/2026-11-26-final/)

055 降低学习率后，对固定的 9 道旧开发题做了 27 个配对，并审计了 510 条仓库请求。底座成功 15 次，055 成功 18 次，其中新增 9 次、退步 6 次。下一阶段在六个独立仓库家族上按既定预算和门槛验收。[055 诊断曲线与决策记录](../experiments/055-retention-sft/results/2026-11-26-diagnostic/)

055 的独立开发中，底座完成 12／180 次，LoRA 完成 21／180 次。净增 5.00 个百分点，配对 95% 区间为 −5.00 至 +16.67 个百分点。多文件任务两组均完成 3／48。 [055 匿名配对结果与调用曲线](../experiments/055-retention-sft/results/2026-11-27-development/)

057 增加了三条经执行验证的多文件修复示范，在六个独立家族上完成 180 个配对。底座修对并提交 27／180 次，057 为 39／180 次，净增 6.67 个百分点，配对 95% 区间为 0.00 至 +16.67 个百分点。收益来自四道单文件题，多文件题两组均为 0／30。057 未通过开发门槛，本轮对封存集的调用数为零。 [057 训练与开发记录](../experiments/057-multifile-correction-sft/)

058 给 79 条训练修复加入实际执行的两次只读查阅，继续使用同一 27B 底座与 412 个监督动作。52 次更新和保存重载检查通过，新的适配器已独立恢复。 [058 训练、试点和冻结记录](../experiments/058-read-history-replay/)

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

042 和 044 都已从各自的私有备份恢复到独立目录，经 CPU 核验逐文件哈希并读取各自的全部 992 个张量。用上述命令恢复这两个适配器时，`--profile` 和 `--receipt` 分别使用对应备份内的 `training-profile.json` 和 `integrity.json`。发布验收另需在匹配底座上独立加载并比对固定输入的生成结果。

## 绑定候选训练记录

后续实验的晋级入口可将通过开发复验的授权、同一候选的适配器和经核验的训练配方交给 `run_approved`，随后用 `audit_approved` 复核原登记的最终测试。打包时成组提供 `--training-protocol`、`--training-profile` 和 `--training-integrity`，核对晋级授权、训练数据、来源清单、配方及适配器哈希。模型卡根据通过核验的记录填写候选名称和训练种子。打包同时重算原最终验收门槛，并核对逐文件许可材料和独立重载回执。[最终验收入口](../experiments/039-release-development/final_campaign.py) · [打包入口](../experiments/039-release-development/release_package.py) · [训练来源校验](../one_gpu_lm/models/release_lineage.py)

041 的九份接续状态使用与 031 相同的五个训练来源和固定版本。已复核工作站快照、上游模型许可清单及十份许可与署名文件的原始字节。打包时按实际入选训练来源的版本复用材料，分发审查继续绑定具体候选、训练数据和拟发布文件。[041 来源回执](../experiments/041-first-edit-continuation/source-notices.json) · [031 来源说明](../experiments/031-correction-mix/release-provenance.md) · [Apache-2.0 分发条款](https://www.apache.org/licenses/LICENSE-2.0#redistribution)

## 核对 ModelScope 上的发布版本

在国际站账号下创建有写权限的访问令牌。在发布电脑的终端安装固定版本的 SDK，运行 `login` 并在隐藏提示中输入令牌，再用 `whoami` 核对账号。浏览器登录和命令行登录各自保存会话，国内站访问令牌仅适用于国内站。[国际站访问令牌](https://www.modelscope.ai/my/myaccesstoken) · [ModelScope 站点说明](https://github.com/modelscope/modelscope-skills#sites-domestic--international)

```bash
python -m pip install modelscope-hub==0.7.0
modelscope-hub --endpoint https://www.modelscope.ai login
modelscope-hub --endpoint https://www.modelscope.ai whoami
```

[发布入口](../experiments/039-release-development/publish_package.py)使用通过验收的打包目录和上述命令行登录态。`check` 核对回执与本地文件。`stage` 创建私有模型仓库，只上传经审核的文件，从实际 Git 提交重新下载并逐文件校验。`publish` 再次比对同一提交后开放访问，匿名下载到新目录。状态保存在打包目录的 `modelscope-release.json`，网络中断后可按原提交恢复。[ModelScope Hub SDK](https://github.com/modelscope/modelscope_hub)

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

### 下载适配器用于推理

用公开发布回执中的 `repo_id` 和完整 `revision` 分别设置 `RELEASE_REPO` 和 `RELEASE_COMMIT`。以下命令从 ModelScope 国际站的指定版本下载两个适配器文件。

```bash
python -m pip install modelscope-hub==0.7.0
modelscope-hub --endpoint https://www.modelscope.ai download \
  "$RELEASE_REPO" adapter_config.json adapter_model.safetensors \
  --revision "$RELEASE_COMMIT" --local-dir ./adapter
```

下载后按模型卡核对两个文件的 SHA-256，再用模型卡中固定的底座版本和加载示例运行推理。

### 核对恢复后的生成结果

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
