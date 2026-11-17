# 030：在相同预算下检验教师思考模式

029 的修复成功数保持 4/9，合格示范由 3 增至 4，未通过采集门槛。剩余五条路径包括不修改、错误修复，以及通过可见检查但没有通过独立训练判分。继续增加流程提醒未必解决修复判断问题；本轮先检验现有教师自身的思考模式，暂不训练新参数。

## 预先固定的对照

- 使用同一固定版本的 Qwen/Qwen3.8-27B，原 BF16 权重、单次加载，不添加适配器或下载外部教师。
- 保留 028 确认的全部 9 个错误补丁状态，按原交替顺序做 18 条新路径，每状态每组一次。它们是已经观察过的训练题，不是独立开发或最终测试。
- 两组都使用 029 的决策提示、可见状态反馈和修改后独立回归检查约束。control 关闭思考；correction 开启模板支持的 `reasoning_effort=low`。
- 两组各有 12 次新调用、每次最多 2048 个生成 token、总计 6144 个生成 token、8192 个输入 token、600 秒代理时间。思考 token、格式错误和拒绝执行均扣预算；没有免费补全、重试、自动修改或代替模型提交。
- 两组单次生成上限均从 029 的 512 调高，因此必须新跑对照，不能将跨实验分数差异归因于思考。共同系统提示明确新预算覆盖历史提示中的限制。
- 单次流水线最长 3 小时，不重复启动，不按结果修改预算。

实际安装的固定版本模板支持思考开关和 low/medium/xhigh 三档；本轮只比较关闭与 low。模型网页未能由检索工具读取，版本与模板能力以已有下载校验、文件哈希及实际 tokenizer 的 CPU 检查为依据。[模板身份记录](preparation/template-identity.json)和[执行协议](execution-protocol.json)共同固定输入。

## 可复用的执行与蒸馏边界

`common/reasoning.py` 从哈希固定的 profile 选择生成模式，不接受任意模板参数。专用推理入口保留 023/024 的原始运行时文件不变。模型思考部分只能留在私有原始回执中；只有唯一、完整结束标记之后的最终动作进入工具解析和后续历史。缺失、重复或嵌套标记都不执行，全部生成仍计费。此规则针对当前固定模板，换模型必须重新验证标记和模板，不假设普遍兼容。

只有独立判分成功、完成可见验证并明确提交的路径，才能进入编码检查。回放逐次验证真实教师输入 token、原始输出 ID、思考分界、执行动作和费用。将最终成功的新修改及其后执行过的动作，重新编码为关闭思考的学生模板：分别记录教师输入哈希和学生输入哈希，保留错误修改及失败反馈作为上下文，禁止截断。思考内容、历史错误动作和早期新失败动作均不作为正向监督目标。

这使“教师如何生成”与“学生如何学习”分离，可用于未来模型迁移。它仍是经过验证的纠错片段，不应包装为从零自主完成的长任务，也不证明学生已经变强。

## 采集门槛与后续

correction 至少覆盖 6 个合格唯一任务、3 个家族、至少 3 个不同于 027 唯一合格任务的题目，相对本轮 control 净增至少 2 条合格示范，修复成功数不下降；18 条路径的身份、预算、源码、请求和编码审计必须全部通过。

过关只允许另行冻结候选训练和开发集评测方案，不直接发布权重。未过关则公开全部对照、失败及成本，停止本轮参数更新。最终测试继续封存，原始题目、命令、轨迹、语料、连接信息和权重不公开。

## 复现入口

先按项目已有步骤准备授权的训练题与错误状态快照，再执行：

```bash
python -m one_gpu_lm.verified.failure_preflight --root . --protocol experiments/030-teacher-reasoning/execution-protocol.json --out private/unique-context-screen.json
python -m one_gpu_lm.verified.failure_watch --root . --protocol experiments/030-teacher-reasoning/execution-protocol.json
python -m one_gpu_lm.verified.failure_audit --root . --protocol experiments/030-teacher-reasoning/execution-protocol.json --out runs/reasoning030-001/audit.json
python -m one_gpu_lm.verified.reasoning_report --results experiments/030-teacher-reasoning/results/2026-11-17-collection --protocol experiments/030-teacher-reasoning/execution-protocol.json
```

协议包含私有输入哈希，不能仅下载公开仓库就假装复现原始轨迹。换自有文档、任务或模型时，应生成新的协议、快照和运行目录，并保留上述检查；不要覆盖历史实验目录。正式启动还要求公开协议和对应提交 CI 已验证。
