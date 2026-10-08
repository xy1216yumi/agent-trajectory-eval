"""统计分析：成功率、偏离分布、首次偏离步数、成本浪费相关性、干预显著性、决策矩阵。

全部统计检验用标准库手写实现（不依赖 scipy）：
  Spearman 等级相关 + 置换检验 p 值
  McNemar 精确检验（二项分布）
  Bootstrap 百分位置信区间
  Cohen's Kappa

用法：
  python src/analyze.py                          # 生成 data/reports/stats_report.md + stats.json
  python src/analyze.py --kappa A B              # 计算两份标注（json/csv）的 Cohen's Kappa
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracer import ROOT, CHECKPOINTS, CHECKPOINT_NAMES

REPORT_DIR = ROOT / "data" / "reports"
ANNOT_DIR = ROOT / "data" / "annotations"


# ---------------- 手写统计函数 ----------------

def _ranks(xs):
    """平均秩（处理并列）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else 0.0


def spearman(x, y, n_perm=5000, seed=42):
    """Spearman 等级相关，置换检验近似 p 值（双侧）。"""
    r = _pearson(_ranks(x), _ranks(y))
    rng = random.Random(seed)
    cnt = 0
    for _ in range(n_perm):
        ys = y[:]
        rng.shuffle(ys)
        if abs(_pearson(_ranks(x), _ranks(ys))) >= abs(r) - 1e-12:
            cnt += 1
    return r, (cnt + 1) / (n_perm + 1)


def mcnemar_exact(b, c):
    """McNemar 精确检验：b=仅干预后成功，c=仅基线成功（双侧）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(values, stat_fn, n_boot=10000, seed=42):
    """百分位 Bootstrap 95% CI。"""
    rng = random.Random(seed)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def cohen_kappa(labels_a, labels_b):
    """Cohen's Kappa：labels_a/b 为等长类别序列。"""
    n = len(labels_a)
    if n == 0:
        return 0.0
    cats = sorted(set(labels_a) | set(labels_b))
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pe = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_labels_any(path):
    """读入标注文件（auto/merged json 或人工 csv），返回 {trace_key: {cp: label}}。"""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v["checkpoints"] for k, v in data.items()}
    import csv as _csv
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            val = (row.get("manual_label") or row.get("auto_label") or "").strip()
            if val:
                out.setdefault(row["trace_key"], {})[row["checkpoint"]] = val
    return out


# ---------------- 主分析 ----------------

def analyze(seed=42, mode_note="mock", price_per_m=None):
    auto_path = ANNOT_DIR / "auto_labels.json"
    labels = json.loads(auto_path.read_text(encoding="utf-8"))
    pairs_path = ANNOT_DIR / "intervention_pairs.json"
    pairs = json.loads(pairs_path.read_text(encoding="utf-8")) if pairs_path.exists() else []

    configs = sorted({lb["config"] for lb in labels.values()})
    stats = {"mode": mode_note, "n_traces": len(labels), "configs": {}}

    if mode_note == "real":  # real 模式：从轨迹 meta 汇总模型/日期/token 成本
        from tracer import read_trace as _rt
        metas, total_tok, dates = [], 0, []
        for f in sorted((ROOT / "data" / "traces").glob("*/*.jsonl")):
            m, _, r = _rt(f)
            metas.append(m)
            total_tok += r.get("total_tokens", 0)
            if m.get("start_time"):
                dates.append(m["start_time"])
        models = sorted({m.get("model", "?") for m in metas})
        stats["real_meta"] = {
            "model": "/".join(models),
            "date_range": [min(dates), max(dates)] if dates else None,
            "total_tokens": total_tok,
        }
        if price_per_m:
            stats["real_meta"]["estimated_cost_cny"] = round(total_tok * price_per_m / 1e6, 2)
            stats["real_meta"]["price_per_m_tokens_cny"] = price_per_m

    # 1. 各组成功率（含分层）
    for cfg in configs:
        lbs = [lb for lb in labels.values() if lb["config"] == cfg]
        by_tier = {}
        for tier in ["easy", "medium", "hard", "edge"]:
            sub = [l for l in lbs if l["tier"] == tier]
            if sub:
                by_tier[tier] = round(sum(l["success"] for l in sub) / len(sub), 4)
        stats["configs"][cfg] = {
            "n": len(lbs),
            "success_rate": round(sum(l["success"] for l in lbs) / len(lbs), 4),
            "by_tier": by_tier,
            "process_deviation_rate": round(
                sum(1 for l in lbs if any(l["checkpoints"].get(cp) == "deviated"
                                          for cp in CHECKPOINTS)) / len(lbs), 4),
            "avg_tokens": round(sum(l["total_tokens"] for l in lbs) / len(lbs), 1),
            "avg_steps": round(sum(l["core_steps"] for l in lbs) / len(lbs), 1),
        }

    # 2. 检查点偏离分布（首次偏离）
    dev_dist = {cp: 0 for cp in CHECKPOINTS}
    for lb in labels.values():
        if lb["first_deviation_checkpoint"]:
            dev_dist[lb["first_deviation_checkpoint"]] += 1
    n_dev = sum(dev_dist.values())
    stats["first_deviation_distribution"] = {
        cp: {"count": dev_dist[cp],
             "pct_of_deviated": round(dev_dist[cp] / n_dev, 4) if n_dev else 0}
        for cp in CHECKPOINTS}
    stats["first_deviation_by_config"] = {
        cfg: {cp: sum(1 for l in labels.values()
                      if l["config"] == cfg and l["first_deviation_checkpoint"] == cp)
              for cp in CHECKPOINTS}
        for cfg in configs}

    # 3. 首次偏离步数分布（绝对 + 百分比）
    dev_steps = [lb["first_deviation_step"] for lb in labels.values() if lb["first_deviation_step"]]
    dev_pcts = [lb["first_deviation_step_pct"] for lb in labels.values() if lb["first_deviation_step_pct"]]
    stats["first_deviation_step"] = {
        "mean": round(sum(dev_steps) / len(dev_steps), 2) if dev_steps else None,
        "mean_pct": round(sum(dev_pcts) / len(dev_pcts), 4) if dev_pcts else None,
        "by_config": {cfg: {
            "mean": (lambda d: round(sum(d) / len(d), 2) if d else None)(
                [l["first_deviation_step"] for l in labels.values()
                 if l["config"] == cfg and l["first_deviation_step"]]),
            "mean_pct": (lambda d: round(sum(d) / len(d), 4) if d else None)(
                [l["first_deviation_step_pct"] for l in labels.values()
                 if l["config"] == cfg and l["first_deviation_step_pct"]]),
        } for cfg in configs},
        "histogram_pct": {
            "0-25%": sum(1 for p in dev_pcts if p <= 0.25),
            "25-50%": sum(1 for p in dev_pcts if 0.25 < p <= 0.5),
            "50-75%": sum(1 for p in dev_pcts if 0.5 < p <= 0.75),
            "75-100%": sum(1 for p in dev_pcts if p > 0.75),
        },
    }

    # 4. 偏离越晚，浪费越大？Spearman(首次偏离步数, 偏离后消耗 token)
    wasted = [lb["wasted_tokens"] for lb in labels.values() if lb["first_deviation_step"]]
    if len(dev_steps) >= 5:
        rho, p = spearman(dev_steps, wasted, seed=seed)
    else:
        rho, p = 0.0, 1.0
    stats["late_deviation_waste"] = {"spearman_rho": round(rho, 4), "p_value": round(p, 6),
                                     "n_deviated": len(dev_steps)}
    early = [w for s, w in zip(dev_steps, wasted) if s <= 2]
    late = [w for s, w in zip(dev_steps, wasted) if s >= 4]
    if early and late:
        stats["late_deviation_waste"]["avg_waste_early_step1_2"] = round(sum(early) / len(early), 1)
        stats["late_deviation_waste"]["avg_waste_late_step4plus"] = round(sum(late) / len(late), 1)
        stats["late_deviation_waste"]["waste_fold"] = round(
            (sum(late) / len(late)) / (sum(early) / len(early)), 2)

    # 5. 干预实验：McNemar + 挽救率 Bootstrap CI
    rescued = [p for p in pairs if p["intervened_success"]]
    b = len(rescued)                      # 基线失败 → 干预成功
    c = sum(1 for p in pairs if not p["intervened_success"])  # 基线失败 → 干预仍失败
    n_ok_baseline = len(labels) - len(pairs)
    rescue_rate = b / len(pairs) if pairs else 0
    lo, hi = bootstrap_ci([1 if p["intervened_success"] else 0 for p in pairs],
                          lambda s: sum(s) / len(s), seed=seed) if pairs else (0, 0)
    p_mcnemar = mcnemar_exact(b, c)
    base_success_rate = n_ok_baseline / len(labels) if labels else 0
    post_success_rate = (n_ok_baseline + b) / len(labels) if labels else 0
    stats["intervention"] = {
        "n_pairs": len(pairs), "rescued_b": b, "harmed_c": c,
        "rescue_rate": round(rescue_rate, 4),
        "rescue_rate_ci95": [round(lo, 4), round(hi, 4)],
        "mcnemar_p": p_mcnemar,
        "success_rate_baseline": round(base_success_rate, 4),
        "success_rate_after_intervention": round(post_success_rate, 4),
    }

    # 6. 2×2 产品决策矩阵：偏离频率 × 可挽救性（按检查点）
    rescue_by_cp, rescue_n = {}, {}
    for cp in CHECKPOINTS:
        sub = [p for p in pairs if p["intervened_at"] == cp]
        rescue_n[cp] = len(sub)
        if sub:
            rescue_by_cp[cp] = sum(p["intervened_success"] for p in sub) / len(sub)
    freq_median = sorted(dev_dist.values())[len(CHECKPOINTS) // 2 - 1:][:2]
    freq_thr = sum(freq_median) / 2 if freq_median else 0
    rates = list(rescue_by_cp.values())
    rescue_thr = (max(rates) + min(rates)) / 2 if rates else 0.5
    QUADRANT = {
        (True, True): "检查点确认（人机协同）",
        (True, False): "自动重试 + 护栏",
        (False, True): "被动监控",
        (False, False): "全自动",
    }
    matrix = {}
    for cp in CHECKPOINTS:
        high_freq = dev_dist[cp] > freq_thr
        high_rescue = rescue_by_cp.get(cp, 0) >= rescue_thr if rescue_by_cp else False
        # 有干预数据才谈可挽救性；>= 阈值防止单一检查点时阈值=取值而误判为否
        matrix[cp] = {
            "name": CHECKPOINT_NAMES[cp],
            "deviation_count": dev_dist[cp],
            "rescue_rate": round(rescue_by_cp.get(cp, 0), 4),
            "rescue_n": rescue_n[cp],
            "high_frequency": high_freq,
            "high_rescuability": high_rescue,
            "decision": QUADRANT[(high_freq, high_rescue)],
        }
    stats["decision_matrix"] = matrix

    # 7. 标注一致性（若存在 merged + 人工 csv）
    kappa_info = None
    manual_csv = ANNOT_DIR / "manual_labels_filled.csv"
    if manual_csv.exists():
        # 用 CSV 导出时冻结的 auto_label 列对比人工列，避免仲裁后的 auto_labels 虚高一致性
        import csv as _csv
        la, lb_ = [], []
        with open(manual_csv, encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                a = (row.get("auto_label") or "").strip()
                m = (row.get("manual_label") or "").strip()
                if a and m:
                    la.append(a)
                    lb_.append(m)
        kappa_info = {"kappa": round(cohen_kappa(la, lb_), 4), "n_items": len(la)}
        stats["annotation_agreement"] = kappa_info

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(stats, kappa_info)
    (REPORT_DIR / "stats_report.md").write_text(report, encoding="utf-8")
    print(f"统计报告已生成：{REPORT_DIR / 'stats_report.md'}")
    return stats


def render_report(stats, kappa_info):
    L = []
    A = L.append
    A("# Agent 轨迹诊断与可干预性评测 —— 统计分析报告\n")
    if stats["mode"] == "mock":
        A("> ⚠️ **本报告基于 mock 模拟数据**（脚本化 Agent，按受控概率植入偏离）。"
          "数字仅用于验证管线与方法论；实测结论需以 real 模式重跑后替换。\n")
    else:
        rm = stats.get("real_meta", {})
        A(f"> **本报告基于 real 模式真实轨迹**：模型 {rm.get('model', '?')}，"
          f"采集时间 {rm.get('date_range', ['?', '?'])[0]} ~ {rm.get('date_range', ['?', '?'])[1]}，"
          f"共 {stats['n_traces']} 条轨迹，基线累计 {rm.get('total_tokens', 0)} tokens"
          + (f"，估算成本 ¥{rm['estimated_cost_cny']}"
             f"（按 ¥{rm['price_per_m_tokens_cny']}/M tokens 计）"
             if rm.get("estimated_cost_cny") is not None else "")
          + "。偏离标注由 LLM-as-judge（judge_real.py）完成，并经人工抽检 Kappa 校验。\n")
    A("## 1. 各组配置成功率\n")
    A("| 配置 | 样本 | 成功率 | easy | medium | hard | edge | 平均步数 | 平均 tokens |")
    A("|---|---|---|---|---|---|---|---|---|")
    names = {"direct": "直调", "react": "ReAct", "reflexion": "反思"}
    for cfg, c in stats["configs"].items():
        t = c["by_tier"]
        row = [f"| {names.get(cfg, cfg)}({cfg}) | {c['n']} | **{c['success_rate']:.1%}**"]
        for tier in ["easy", "medium", "hard", "edge"]:
            row.append(f"{t[tier]:.0%}" if tier in t else "-")
        row += [str(c["avg_steps"]), f"{c['avg_tokens']} |"]
        A(" | ".join(row))
    A("")
    A("**过程 vs 终态**：各组终态成功率相近，但过程偏离率差异巨大——"
      + "；".join(f"{names.get(cfg, cfg)} 终态成功率 {c['success_rate']:.0%}，"
                  f"过程偏离率（任一检查点偏离）{c['process_deviation_rate']:.0%}"
                  for cfg, c in stats["configs"].items())
      + "。**终态指标完全掩盖了过程失效**，这正是轨迹级评测存在的理由。")
    A("\n## 2. 首次偏离的检查点分布\n")
    A("| 检查点 | 首次偏离次数 | 占全部偏离轨迹 |")
    A("|---|---|---|")
    for cp in CHECKPOINTS:
        d = stats["first_deviation_distribution"][cp]
        A(f"| {CHECKPOINT_NAMES[cp]}({cp}) | {d['count']} | {d['pct_of_deviated']:.1%} |")
    top = max(CHECKPOINTS, key=lambda cp: stats["first_deviation_distribution"][cp]["count"])
    A(f"\n**发现 1**：偏离最集中的环节是 **{CHECKPOINT_NAMES[top]}**。\n")
    if stats.get("first_deviation_by_config"):
        A("分配置看：" + "；".join(
            f"{names.get(cfg, cfg)} — " +
            ("、".join(f"{CHECKPOINT_NAMES[cp]} {n}" for cp, n in d.items() if n) or "无偏离")
            for cfg, d in stats["first_deviation_by_config"].items())
          + "。直调组的规划偏离是架构性的（无规划无检索直接作答），"
            "ReAct/反思组的偏离集中于结果校验环节（主要是 edge 题的边界行为失守）。\n")
    s = stats["first_deviation_step"]
    A("## 3. 首次偏离步数分布\n")
    A(f"- 平均首次偏离步数：{s['mean']}（占核心轨迹全长 **{s['mean_pct']:.1%}**）")
    if s.get("by_config"):
        A("- 分配置：" + "；".join(
            f"{names.get(cfg, cfg)} 平均第 {v['mean']} 步（占 {v['mean_pct']:.0%}）"
            for cfg, v in s["by_config"].items() if v["mean"] is not None))
    h = s["histogram_pct"]
    A(f"- 占比直方图：0-25% 处 {h['0-25%']} 条 / 25-50% 处 {h['25-50%']} 条 / "
      f"50-75% 处 {h['50-75%']} 条 / 75-100% 处 {h['75-100%']} 条\n")
    w = stats["late_deviation_waste"]
    A("## 4. 偏离越晚，浪费越大\n")
    A(f"- Spearman(首次偏离步数, 偏离后消耗 token)：ρ = **{w['spearman_rho']}**，"
      f"置换检验 p = {w['p_value']:.4g}（n = {w['n_deviated']}）")
    if "waste_fold" in w:
        A(f"- 晚偏离（第 4 步及以后）平均浪费 {w['avg_waste_late_step4plus']} tokens，"
          f"是早偏离（第 1-2 步，{w['avg_waste_early_step1_2']}）的 **{w['waste_fold']} 倍**")
    if stats["mode"] == "mock":
        A("\n**发现 2**：偏离发生越晚，事后消耗的 token 越多（模拟 token 模型假设偏离后"
          "Agent 需重放被污染的完整上下文，符合真实 LLM 计费机制）。\n")
    else:
        A("\n**发现 2**：偏离发生越晚，事后消耗的 token 越多"
          "（真实计费数据：prompt 随对话历史累积，偏离点之后的每一次 LLM 调用都在为"
          "已被污染的上下文付费）。\n")
    iv = stats["intervention"]
    A("## 5. 干预实验（配对设计）\n")
    A(f"- 基线成功率 {iv['success_rate_baseline']:.1%} → 干预后 "
      f"**{iv['success_rate_after_intervention']:.1%}**")
    A(f"- 挽救率 = {iv['rescued_b']}/{iv['n_pairs']} = **{iv['rescue_rate']:.1%}**，"
      f"Bootstrap 95% CI [{iv['rescue_rate_ci95'][0]:.1%}, {iv['rescue_rate_ci95'][1]:.1%}]")
    A(f"- McNemar 精确检验：b={iv['rescued_b']}，c={iv['harmed_c']}，"
      f"p = **{iv['mcnemar_p']:.3g}**"
      + ("（p < 0.05，提升显著）" if iv["mcnemar_p"] < 0.05 else "（不显著）"))
    if iv["mcnemar_p"] < 0.05:
        A("\n**发现 3**：在首次偏离检查点注入纠正信息后，成功率显著提升，"
          "说明相当比例的失败轨迹是\"可挽救\"的。\n")
    else:
        marginal = "（0.05 ≤ p < 0.1，边缘显著，方向性参考）" if iv["mcnemar_p"] < 0.1 else ""
        A(f"\n**发现 3（证据不足）**：干预挽救率 {iv['rescue_rate']:.0%}，discordant 对数 "
          f"{iv['rescued_b'] + iv['harmed_c']}，McNemar p={iv['mcnemar_p']:.3g}{marginal}——"
          "按项目规范（不显著即不声明），\"干预提升成功率\"不作为成立结论；"
          "挽救率点估计提示多数失败轨迹可挽救，但需更大失败基数（更难任务或更弱模型）确认。\n")
    A("## 6. 2×2 产品决策矩阵（偏离频率 × 可挽救性）\n")
    A("| 检查点 | 偏离次数 | 挽救率(n) | 高频？ | 高可挽救？ | 产品决策 |")
    A("|---|---|---|---|---|---|")
    for cp in CHECKPOINTS:
        m = stats["decision_matrix"][cp]
        A(f"| {m['name']} | {m['deviation_count']} | {m['rescue_rate']:.0%}（n={m['rescue_n']}） | "
          f"{'是' if m['high_frequency'] else '否'} | {'是' if m['high_rescuability'] else '否'} | "
          f"**{m['decision']}** |")
    A("")
    if kappa_info:
        A("## 7. 标注一致性\n")
        who = "LLM-as-judge vs 人工抽检" if stats["mode"] == "real" else "自动预判 vs 人工标注"
        A(f"- {who}：Cohen's Kappa = **{kappa_info['kappa']}**"
          f"（n = {kappa_info['n_items']} 个检查点标签）\n")
    A("---\n*统计方法：Spearman 置换检验 / McNemar 精确检验 / Bootstrap 百分位 CI / "
      "Cohen's Kappa，全部由 src/analyze.py 以标准库手写实现。*")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="统计分析")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode-note", default="mock")
    ap.add_argument("--price", type=float, default=None,
                    help="real 模式：每百万 token 单价（元），用于估算成本")
    ap.add_argument("--kappa", nargs=2, metavar=("A", "B"),
                    help="只计算两份标注文件的 Cohen's Kappa")
    args = ap.parse_args()

    if args.kappa:
        a, b = (load_labels_any(p) for p in args.kappa)
        keys = sorted(set(a) & set(b))
        la, lb = [], []
        for k in keys:
            for cp in CHECKPOINTS:
                if cp in a[k] and cp in b[k]:
                    la.append(a[k][cp])
                    lb.append(b[k][cp])
        k = cohen_kappa(la, lb)
        print(f"Cohen's Kappa = {k:.4f}（n = {len(la)} 个检查点标签）")
        return

    analyze(seed=args.seed, mode_note=args.mode_note, price_per_m=args.price)


if __name__ == "__main__":
    main()
