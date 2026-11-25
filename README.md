# One GPU LM

在一张工作站 GPU 上训练和评测编码模型。用同一批仓库任务、同一套工具和同样的预算对照底座与 LoRA，公开修复成功率、退步和实测卡时。

*Train and evaluate coding models on one workstation GPU. Rebuild the paired results from public measurements.*

[CPU 快速复算](#cpu-快速复算) · [27B 发布进度](docs/modelscope-release.md) · [实验索引](docs/experiment-index.md) · [如何贡献](#如何贡献)

## 公开结果

053 在六个新的仓库家族上测试了 60 道题，每组各跑三次。底座修对并提交 **36/180（20%）**，LoRA 为 **54/180（30%）**。配对 95% 区间为 **−1.67 至 +23.33 个百分点**，这一轮未通过开发门槛。曲线记录首次改码和修对并提交的进度。[逐题结果、曲线与卡时](experiments/053-post-edit-recheck/results/2026-11-25-development/)

![053 底座和 LoRA 随调用次数累计首次改码及成功提交的曲线](experiments/053-post-edit-recheck/results/2026-11-25-development/call-curves.svg)

054 在一张 RTX PRO 6000 Blackwell 96 GB 上对 27B 底座做 LoRA 微调。50 次更新的训练循环用时 **16.1 分钟**，实测峰值显存约 **90 GiB**。首个 27B 适配器仍待质量验收和发布。[训练曲线与硬件记录](experiments/054-expanded-repair-sft/results/2026-11-25-training/) · [开发对照登记](experiments/054-expanded-repair-sft/development-registration.json)

## CPU 快速复算

用公开的 053 匿名测量重算通过率、配对区间和开发门槛，核对逐次结果与曲线数值。使用 Python 3.10 或以上版本的标准库即可。

```bash
git clone https://github.com/CaptainGreenskin/one-gpu-lm.git
cd one-gpu-lm
python3 -B scripts/verify_development_report.py \
  experiments/053-post-edit-recheck/results/2026-11-25-development
```

输出包含 `"base_successes": 36`、`"candidate_successes": 54` 和 `"development_gate_passed": false`，以及三份逐项核对过的数值文件。053 的任务快照、推理和训练记录由实验方保留，下面的入口可以用自己的任务实际训练和评测。

## 换成自己的工作

| 可以复用什么 | 起点 |
| --- | --- |
| 在自己的中文文档上测过拟合和训练遍数 | [训练与留出文档各放一个目录](experiments/007-document-curves/#用自己的文档跑) · [独立测试上的早停对照](experiments/008-online-early-stop/) |
| 测 7B 配置在不同上下文长度下的显存和吞吐 | [全参数更新、LoRA 与重算的对照](experiments/010-7b-context-scaling/) |
| 用自己的仓库任务比较底座和适配器 | [任务格式、离线判分与恢复](repo_lab/) · [27B 训练操作手册](docs/27b-training.md) · [换底座的检查项](docs/model-upgrades.md) |

## 如何贡献

先试一次 CPU 复算。遇到差异时，提交操作系统、Python 版本和命令输出。也欢迎用自己的显卡、模型或文档复跑，分享运行命令、版本、耗时和可公开的指标。[贡献指南](CONTRIBUTING.md) · [提交复跑记录](https://github.com/CaptainGreenskin/one-gpu-lm/issues/new)

原创代码采用 MIT 许可，模型和数据遵循各自的许可。训练项目与知识笔记独立维护。[代码许可](LICENSE) · [InfraGuide 知识项目](https://github.com/CaptainGreenskin/infra-guide)
