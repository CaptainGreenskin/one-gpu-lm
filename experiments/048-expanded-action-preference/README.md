# 048 · 合并已审计的首次动作偏好样本

训练入口接收 045 已审核的样本和 047 完整采集的样本。它逐批核对源码、剩余预算、两条首步生成回执和真实 tokenizer 编码，保留每批的控制器、审计与数据摘要。同一训练任务或同一 token 前缀只出现一次。 [045 采集](../045-first-action-forks/) · [047 采集](../047-expanded-action-forks/)

使用原来的 rank-8 LoRA 训练入口。冻结候选时提供 `--expanded-collection-protocol` 和 `--expanded-results-publication`，同时提供原有 045 协议和独立开发集的身份。数据准入依据 047 的完整审计和公开结果核验回执。 [训练入口](../../one_gpu_lm/verified/decision_training.py) · [样本核验](../../one_gpu_lm/verified/expanded_preference.py)

训练审计通过后，报表按每次优化器更新汇总样本数、偏好损失、相对底座的偏好边际、梯度范数和显存峰值，同时保留可核验的 GPU 采样与续跑记录。 [报表生成器](report.py)

独立开发评测沿用同一 27B 底座、原始开场、三轮重复和相同任务预算。门槛继续要求配对成功率至少提升 10 个百分点、95% 区间下界大于零，每个家族的下降至多 10 个百分点。达标后进入封存的最终验收。 [发布方案](../../docs/modelscope-release.md)
