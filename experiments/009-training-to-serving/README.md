# 009 · 单卡从训练到推理服务

一条命令完成数据准备、训练与早停、测试门禁、导出、推理测量和 HTTP 部署验收。沿用 008 的两份小语料及训练预算，预先选 Vue、种子 101 的早停权重演示部署。模型按固定标识选取，与测试排名无关。这个 85.6M BPE 模型提供文本续写。另设可选 7B 容量测试，测量这一张卡能实际更新多少参数、每秒处理多少 token 和峰值显存。

在仓库根目录、使用 [008 的 Linux/PyTorch 环境](../008-online-early-stop/) 运行。输出目录必须是新目录；完整训练包含 18 次独立运行。

```bash
exp=experiments/009-training-to-serving
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
python "$exp/pipeline.py" --out runs/system
# 若本机已有跑完测试的 008 权重，跳过重新训练，也可同时实测 7B：
python "$exp/pipeline.py" --from-run runs/008 --out runs/system-reuse --seven-b
```

- **训练吞吐。** 分语料和策略，报告实读 BPE token 数除以权重更新秒数，以及除以整段占卡秒数。后者包含初始化、在线验证和保存。
- **模型导出。** 在本机生成 `bundle/{weights.pt,tokenizer.json,bundle.json}`，记录层配置、训练选择、权重和分词器 SHA-256。加载时验证文件。当前运行时代码来自本仓库。
- **推理基准。** 使用同一 checkpoint、BF16、相同输入 token 流。比较全前缀重算与每个请求独立的静态 KV cache。批大小 1/8，提示长度 32/128，各测 32 个增量解码步。计时前热身，两种方式交替顺序，各测五次。prefill 包含首次 logits 与 argmax，decode 单独记录批量 token/s 与每步耗时，两种方法都只为最后一个位置算输出头。固定 token 流用于控制工作量，贪心续写另行核验。显存记录 PyTorch 峰值 allocated。FP32 核验覆盖多 token 后缀、单 token 解码与位置偏移；BF16 另报舍入差异。
- **服务。** 模型启动时加载一次，API 串行服务，每个请求有独立缓存。默认绑定本机回环地址。流水线实际启动 HTTP 进程，验收健康检查、续写和超长输入，再释放设备。要持续提供服务，另行启动下方命令。
- **可选 7B。** 一个随机初始化的稠密 GPT，32 层、32 个注意力头、隐藏宽度 4096、前馈宽度 18048、32,000 项词表，输入输出权重共享。对合成 token 用两种方式各实训 4 次更新（第 1 次热身，后三次计吞吐），批大小 1、长度 128，无 CPU 卸载。BF16 冻结主干的 QKV LoRA 使用 rank 8、FP32 适配器和 AdamW；全参数模式使用 BF16 参数和梯度、因子化状态的 Adafactor。两种模式各核对梯度覆盖、有限损失及真实参数改动。小规模容量测试只回答这个配置能否跑通及其短时性能。[Adafactor 论文](https://arxiv.org/abs/1804.04235)

```bash
python experiments/009-training-to-serving/serve.py --bundle runs/system/bundle --port 8080 --device cuda
# 另开一个终端：
curl -s http://127.0.0.1:8080/health
curl -s -H 'Content-Type: application/json' -d '{"prompt":"计算属性","max_new_tokens":16}' http://127.0.0.1:8080/generate
```

权重、编码缓存和文本留在本地。公开证据包括模型包清单、硬件和软件信息、逐次耗时、缓存一致性检查、HTTP 状态码和优化器步数。
