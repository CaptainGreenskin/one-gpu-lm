"""Audit complete recomputation pairs, aggregate process repeats, and plot."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def compare(a, b, protocol):
    if a["status"] != "ok" or b["status"] != "ok":
        return dict(valid=False, reason="incomplete_pair")
    for key in ("mode", "sequence", "repeat", "batch", "config", "base_parameters", "trainable_parameters",
                "input_sha256", "initial_probe_sha256", "protocol_sha256"):
        if a[key] != b[key]:
            return dict(valid=False, reason=f"mismatch_{key}")
    expected = protocol["warmup_steps"] + protocol["measured_steps"]
    if len(a["records"]) != expected or len(b["records"]) != expected:
        return dict(valid=False, reason="missing_steps")
    loss_gap = max(abs(x["loss"] - y["loss"]) for x, y in zip(a["records"], b["records"]))
    grad_gap = max(abs(x["grad_norm"] - y["grad_norm"]) / max(abs(x["grad_norm"]), 1e-12)
                   for x, y in zip(a["records"], b["records"]))
    valid = loss_gap <= protocol["comparison_loss_atol"] and grad_gap <= protocol["comparison_grad_norm_rtol"]
    return dict(valid=valid, reason="ok" if valid else "numerical_mismatch", max_loss_gap=loss_gap,
                max_grad_relative_gap=grad_gap, throughput_ratio_checkpoint_over_plain=b["tokens_per_second"] / a["tokens_per_second"],
                allocated_reduction_fraction=1 - b["peak_allocated_bytes"] / a["peak_allocated_bytes"])


def summarize(root):
    protocol_bytes = (root / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    campaign = json.loads((root / "campaign.json").read_bytes())
    rows, audits, groups = [], [], []
    index = {}
    status_counts = {}
    for rec in campaign["records"]:
        r = json.loads((root / (rec["name"] + ".json")).read_bytes())
        assert r["status"] == rec["status"]
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        key = (r["mode"], r["sequence"], r["repeat"], r["checkpoint"])
        assert key not in index
        index[key] = r
        if r["status"] == "ok":
            assert r["returncode"] == 0
            assert r["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
            assert r["config"] == protocol["model_config"] and r["batch"] == protocol["batch"]
            assert r["base_parameters"] == 7028273152
            assert r["trainable_elements_with_gradient"] == r["trainable_parameters"]
            assert all(x > 0 for x in r["probe_max_abs_updates"])
            measured = [x for x in r["records"] if not x["warmup"]]
            assert len(measured) == protocol["measured_steps"]
            expected = r["batch"] * r["sequence"] * len(measured) / sum(x["seconds"] for x in measured)
            assert math.isclose(expected, r["tokens_per_second"], rel_tol=1e-12)
        rows.append({k: r.get(k) for k in ("mode", "sequence", "repeat", "checkpoint", "status", "tokens_per_second",
                    "peak_allocated_bytes", "peak_reserved_bytes", "process_wall_seconds", "initialization_seconds",
                    "free_before_bytes", "allocator_cap_bytes")})
    for mode in protocol["modes"]:
        for seq in protocol["sequences"]:
            valid = []
            for repeat in range(protocol["repeats"]):
                a, b = [index.get((mode, seq, repeat, ac)) for ac in (False, True)]
                pair = compare(a, b, protocol) if a and b else dict(valid=False, reason="missing_trial")
                audits.append(dict(mode=mode, sequence=seq, repeat=repeat, **pair))
                if pair["valid"]:
                    valid.append((a, b))
            for ac in (False, True):
                selected = [pair[int(ac)] for pair in valid]
                if not selected:
                    continue
                speed = [r["tokens_per_second"] for r in selected]
                groups.append(dict(mode=mode, sequence=seq, checkpoint=ac, complete_pairs=len(valid),
                                   tokens_per_second_median=statistics.median(speed), tokens_per_second_min=min(speed),
                                   tokens_per_second_max=max(speed),
                                   peak_allocated_gib_median=statistics.median(r["peak_allocated_bytes"] / 2**30 for r in selected),
                                   peak_allocated_gib_max=max(r["peak_allocated_bytes"] / 2**30 for r in selected),
                                   process_wall_seconds_median=statistics.median(r["process_wall_seconds"] for r in selected)))
    with (root / "telemetry.csv").open() as stream:
        telemetry_rows = list(csv.DictReader(stream))
    telemetry = dict(samples=len(telemetry_rows), nominal_interval_seconds=2)
    for key in ("memory_used_mib", "utilization_percent", "temperature_c", "power_w", "sm_clock_mhz"):
        values = []
        for row in telemetry_rows:
            try:
                values.append(float(row[key]))
            except (ValueError, KeyError):
                pass
        if values:
            telemetry[key] = dict(mean=statistics.mean(values), min=min(values), max=max(values))
    return dict(protocol=protocol, campaign={k: v for k, v in campaign.items() if k != "records"},
                status_counts=status_counts, valid_pairs=sum(a["valid"] for a in audits), planned_pairs=len(audits),
                pairs=audits, groups=groups, telemetry=telemetry), rows


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    labels = {"full-adafactor": "Full BF16 / Adafactor", "lora-adamw": "BF16 backbone / LoRA + AdamW"}
    for col, mode in enumerate(summary["protocol"]["modes"]):
        for ac, color in ((False, "#2563eb"), (True, "#d97706")):
            groups = sorted((g for g in summary["groups"] if g["mode"] == mode and g["checkpoint"] == ac), key=lambda g: g["sequence"])
            x = [g["sequence"] for g in groups]
            y = [g["tokens_per_second_median"] for g in groups]
            label = "Block recomputation" if ac else "No recomputation"
            axes[0, col].plot(x, y, "o-", color=color, label=label)
            axes[0, col].fill_between(x, [g["tokens_per_second_min"] for g in groups], [g["tokens_per_second_max"] for g in groups], color=color, alpha=.12)
            axes[1, col].plot(x, [g["peak_allocated_gib_median"] for g in groups], "o-", color=color, label=label)
        axes[0, col].set_title(labels[mode])
        for row in range(2):
            ax = axes[row, col]
            ax.set_xscale("log", base=2)
            ax.set_xticks(summary["protocol"]["sequences"], labels=[str(s) for s in summary["protocol"]["sequences"]])
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
        axes[0, col].set_ylabel("Training token/s (median; band = min/max)")
        axes[1, col].set_ylabel("Peak PyTorch allocated (GiB, median)")
        axes[1, col].set_xlabel("Sequence tokens, batch = 1")
        axes[0, col].legend(fontsize=8)
    fig.suptitle("7.028B on RTX PRO 6000: context length vs recomputation\nSynthetic tokens; 2 warmup + 6 measured updates per process; 3 process repeats", fontsize=12)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def markdown(summary):
    lines = ["# 010 · 7B 上下文与重算实测", "", "同一随机初始化的 7,028,273,152 参数 GPT，batch 1，合成 token。表格仅纳入通过数值与输入审计的完整配对，报告进程重复中位数。", "",
             f"完成状态：`{summary['campaign']['state']}`；记录 {sum(summary['status_counts'].values())} 个进程，状态 `{summary['status_counts']}`，有效配对 {summary['valid_pairs']}/{summary['planned_pairs']}。",
             f"实际起止（UTC）：{summary['campaign']['started_utc']} 至 {summary['campaign']['finished_utc']}；总计 {summary['campaign']['wall_seconds']:.1f} 秒。", "",
             "![上下文长度与重算曲线](context-scaling.png)", "", "| 更新方式 | 长度 | 重算 | 完整配对 | token/s | 峰值 allocated GiB |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for g in summary["groups"]:
        lines.append(f"| {g['mode']} | {g['sequence']} | {'开' if g['checkpoint'] else '关'} | {g['complete_pairs']} | {g['tokens_per_second_median']:.1f} | {g['peak_allocated_gib_median']:.2f} |")
    valid = [p for p in summary["pairs"] if p["valid"]]
    if valid:
        lines += ["", f"配对最大逐步 loss 差 {max(p['max_loss_gap'] for p in valid):.8g}；最大梯度范数相对差 {max(p['max_grad_relative_gap'] for p in valid):.8g}。两种实现均有所有可训练参数的梯度和可测参数改动。"]
    lines += ["", "原始进程 JSON、配对审计 `summary.json`、扁平表 `runs.csv`、设备采样 `telemetry.csv` 与源码 SHA-256 `sources.json` 一并公开。峰值 allocated 不含驱动/其他进程，设备采样是离散点，不替代分配器峰值。耗时包含检查与裁剪，未包含合成数据生成与上传。",
              "", "短测重复的范围不是统计置信区间。BF16 参数与 Adafactor 没有 FP32 主权重；本结果不证明该精度方案长训稳定，也不代表现成 7B 架构的吞吐。没有把随机模型作为中文能力模型发布。", "",
              "复现与重画见[实验入口](../../README.md)。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = summarize(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.out / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(summary, args.out / "context-scaling.png")
    (args.out / "README.md").write_text(markdown(summary))
    print(json.dumps(dict(valid_pairs=summary["valid_pairs"], planned_pairs=summary["planned_pairs"], statuses=summary["status_counts"])))


if __name__ == "__main__":
    main()
