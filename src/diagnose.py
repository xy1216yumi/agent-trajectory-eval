"""按《偏离判据手册》对每条轨迹做规则化自动预判（逐检查点打标）。

用法：
  python src/diagnose.py                        # 自动预判 → data/annotations/auto_labels.json
  python src/diagnose.py --export-manual        # 导出人工标注 CSV 模板
  python src/diagnose.py --merge <人工标注.csv>  # 合并人工标注 → merged_labels.json

注意：自动预判只是启发式初筛，金标准以人工标注为准（见 docs/02-偏离判据手册.md）。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_agent import load_tasks
from tracer import ROOT, CHECKPOINTS, read_trace

ANNOT_DIR = ROOT / "data" / "annotations"


def _common_substr_len(a, b):
    """两字符串最长公共子串长度（判据规则的字符串匹配工具）。"""
    best = 0
    for i in range(len(a)):
        for j in range(i + 4, len(a) + 1):
            if a[i:j] in b:
                best = max(best, j - i)
            else:
                break
    return best


def diagnose_trace(path, task):
    """对一条轨迹逐检查点应用判据，返回标签 dict。"""
    meta, events, result = read_trace(path)
    steps = [e for e in events if e["event"] == "step"]
    tools = [e for e in events if e["event"] == "tool_call"]
    core_steps = [s for s in steps if s["action"] != "纠偏重试"]
    n_core = len(core_steps)
    evidence_by_id = {}
    for tc in tools:
        for ev in tc.get("evidence", []):
            evidence_by_id[ev["id"]] = ev["fact"]

    labels = {}
    details = []
    first = None  # (step_index, checkpoint, detail)

    for s in core_steps:
        cp = s["checkpoint"]
        if cp in labels and labels[cp] == "deviated":
            continue
        obs = s.get("observable", {})
        dev = None

        if cp == "perception":
            got = obs.get("conditions_extracted", [])
            for cond in task["key_conditions"]:
                if cond not in got:
                    dev = f"关键条件缺失或被曲解：{cond}"
                    break
        elif cp == "planning":
            plan = obs.get("plan", [])
            required = [r["name"] for r in task["required_steps"]]
            missing = [r for r in required if r not in plan]
            if missing:
                dev = f"计划缺少必要步骤：{missing[0]}"
            else:
                idx = [plan.index(r) for r in required]
                if idx != sorted(idx):
                    dev = "计划步骤顺序与标准流程不符"
        elif cp == "tool_use":
            tc = next((t for t in tools if t["step_index"] == s["step_index"]), None)
            if tc is None:
                dev = "需要检索核实但未调用工具（该用未用）"
            elif tc.get("expected_tool") and tc["tool"] != tc["expected_tool"]:
                dev = f"调用错误工具：应 {tc['expected_tool']}，实际 {tc['tool']}"
            elif tc["tool"] == "search" and _common_substr_len(
                    tc.get("args", {}).get("query", ""), task["question"]) < 4:
                dev = "检索参数与问题不匹配（参数错误）"
        elif cp == "verification":
            citations = obs.get("citations", [])
            for c in citations:
                sid = c.get("source_id")
                if sid not in evidence_by_id:
                    dev = f"引用不存在：{sid} 不在检索证据中"
                    break
                if _common_substr_len(c.get("claim", ""), evidence_by_id[sid]) < 4:
                    dev = f"结论与证据不符：引用 {sid} 的表述与证据内容无对应"
                    break
            if dev is None and task["tier"] == "edge":
                if obs.get("final_behavior") != task.get("expected_behavior"):
                    dev = (f"边界题应 {task.get('expected_behavior')}，"
                           f"实际 {obs.get('final_behavior')}")
        if dev:
            labels[cp] = "deviated"
            details.append({"checkpoint": cp, "step": s["step_index"], "rule": dev})
            if first is None:
                first = (s["step_index"], cp, dev)
        else:
            labels.setdefault(cp, "pass")

    for cp in CHECKPOINTS:
        labels.setdefault(cp, "not_reached")

    return {
        "task_id": meta.get("task_id", task["id"]),
        "config": meta.get("config", ""),
        "tier": meta.get("tier", task["tier"]),
        "success": bool(result.get("success")),
        "checkpoints": labels,
        "first_deviation_checkpoint": first[1] if first else None,
        "first_deviation_step": first[0] if first else None,
        "first_deviation_step_pct": round(first[0] / n_core, 4) if first and n_core else None,
        "core_steps": n_core,
        "total_tokens": result.get("total_tokens", 0),
        "wasted_tokens": result.get("wasted_tokens_after_deviation", 0),
        "cost_waste_ratio": result.get("cost_waste_ratio", 0.0),
        "details": details,
    }


def run_diagnose(traces_dir=None, tasks=None):
    traces_dir = Path(traces_dir) if traces_dir else ROOT / "data" / "traces"
    tasks = tasks or load_tasks()
    task_map = {t["id"]: t for t in tasks}
    out = {}
    for f in sorted(traces_dir.glob("*/*.jsonl")):
        meta, _, _ = read_trace(f)
        task = task_map.get(meta.get("task_id"))
        if task is None:
            continue
        key = f"{meta.get('config', f.parent.name)}/{meta['task_id']}"
        out[key] = diagnose_trace(f, task)
    return out


def export_manual(labels, path):
    rows = [["trace_key", "task_id", "config", "checkpoint", "auto_label", "manual_label"]]
    for key, lb in sorted(labels.items()):
        for cp in CHECKPOINTS:
            rows.append([key, lb["task_id"], lb["config"], cp,
                         lb["checkpoints"][cp], ""])
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def merge_manual(labels, csv_path, out_path):
    """人工标注优先，留空回退自动预判。"""
    manual = {}
    with open(csv_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            val = (row.get("manual_label") or "").strip()
            if val:
                manual[(row["trace_key"], row["checkpoint"])] = val
    merged = json.loads(json.dumps(labels, ensure_ascii=False))
    n_override = 0
    for key, lb in merged.items():
        for cp in CHECKPOINTS:
            v = manual.get((key, cp))
            if v:
                if v != lb["checkpoints"][cp]:
                    n_override += 1
                lb["checkpoints"][cp] = v
        # 依据合并后标签重算首次偏离检查点
        order = {cp: i for i, cp in enumerate(CHECKPOINTS)}
        deviated = [cp for cp in CHECKPOINTS if lb["checkpoints"][cp] == "deviated"]
        lb["first_deviation_checkpoint"] = min(deviated, key=order.get) if deviated else None
    Path(out_path).write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return out_path, n_override


def main():
    ap = argparse.ArgumentParser(description="轨迹偏离自动预判")
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--export-manual", action="store_true")
    ap.add_argument("--merge", default=None, help="人工标注 CSV 路径")
    args = ap.parse_args()

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = run_diagnose(args.traces_dir)
    auto_path = ANNOT_DIR / "auto_labels.json"
    auto_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    n_dev = sum(1 for lb in labels.values() if lb["first_deviation_checkpoint"])
    print(f"自动预判完成：{len(labels)} 条轨迹，{n_dev} 条存在偏离 → {auto_path}")

    if args.export_manual:
        p = export_manual(labels, ANNOT_DIR / "manual_labels_template.csv")
        print(f"人工标注模板已导出：{p}（填写 manual_label 列：pass/deviated/not_reached）")
    if args.merge:
        out, n = merge_manual(labels, args.merge, ANNOT_DIR / "merged_labels.json")
        print(f"已合并人工标注（覆盖 {n} 处）→ {out}")


if __name__ == "__main__":
    main()
