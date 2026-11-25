# 054 · 扩充可验证的修复样本

053 的独立开发对照通过率上升了 10 个百分点，配对 95% 区间下限仍低于零。这一轮从同一个 27B 底座初始化新 LoRA，训练数据覆盖 79 个真实修复案例和 394 条实际执行的训练动作。其中有 32 次对部分修复运行的回归测试和 22 次看到失败后执行的修改。[053 开发结果](https://github.com/CaptainGreenskin/one-gpu-lm/tree/ec9a086a6761aad7f5d95a6d0efdc260f2246f3e/experiments/053-post-edit-recheck/results/2026-11-25-development)

沿用原来的 LoRA 配置、优化器和一遍训练。数据量增大后，更新次数从 41 增至 50。开发验收使用独立的新仓库任务，沿用推理流程、每题调用和 token 预算以及既定门槛。最终测试继续封存。

[训练设计](design.json) · [训练登记](training-registration.json)
