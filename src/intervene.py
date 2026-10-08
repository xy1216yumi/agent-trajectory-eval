"""干预实验（配对设计）：对无干预失败轨迹，在首次偏离检查点注入纠正信息后重跑。

用法：
  python src/intervene.py --mode mock --seed 42
  python src/intervene.py --mode real          # 需先桥接 API 环境变量，且已跑过 judge_real.py

mock 模式：被纠正检查点的偏离概率清零，其余检查点偏离概率 ×0.25。
real 模式：把 judge 诊断出的首次偏离点对应的正确信息（关键条件/正确步骤/
正确证据/正确结论或应拒绝判定）以"检查点确认"形式注入 prompt，按同配置重跑，
再由 judge_real 重判。
输出：data/traces_intervened/{config}/*.jsonl + data/annotations/intervention_pairs.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose import diagnose_trace
from mock_agent import GROUP_CONFIGS, load_tasks, run_mock_task
from tracer import ROOT


def build_correction(task, checkpoint):
    """按首次偏离检查点生成"检查点确认"纠正信息。"""
    if checkpoint == "perception":
        return ("本题的关键条件为：\n- " + "\n- ".join(task["key_conditions"]) +
                "\n请逐条核对，不要漏读或误读。")
    if checkpoint == "planning":
        steps = " → ".join(s["name"] for s in task["required_steps"])
        return f"请严格按以下流程执行：{steps}。不要跳步、不要颠倒顺序。"
    if checkpoint == "tool_use":
        facts = "\n".join(f"- {f}" for f in task.get("evidence_facts", []))
        return ("必须使用 search 工具检索、并用 citation_check 核验每条引用后再下结论。"
                f"检索系统就本题返回的证据为：\n{facts}")
    if checkpoint == "verification":
        base = f"经人工核对，本题的正确结论为：{task['ground_truth']}"
        if task["tier"] == "edge":
            base += f"\n本题期望行为为 {task.get('expected_behavior')}：请据此作答，不要给出武断结论。"
        else:
            base += "\n引用的每条来源必须真实出现在检索结果中，结论必须与证据一致。"
        return base
    raise ValueError(f"未知检查点：{checkpoint}")


def run_intervention_mock(seed=42, strength=0.25):
    auto_path = ROOT / "data" / "annotations" / "auto_labels.json"
    if not auto_path.exists():
        raise RuntimeError("未找到 auto_labels.json，请先运行 python src/diagnose.py")
    labels = json.loads(auto_path.read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in load_tasks()}

    pairs = []
    out_dir = ROOT / "data" / "traces_intervened"
    for key, lb in sorted(labels.items()):
        if lb["success"]:
            continue  # 只对无干预失败轨迹做干预
        task = tasks[lb["task_id"]]
        cfg = lb["config"]
        intervene_at = lb["first_deviation_checkpoint"]
        out = out_dir / cfg
        out.mkdir(parents=True, exist_ok=True)
        run_mock_task(task, cfg, seed + 100000, out_dir=out,
                      intervene_at=intervene_at, intervention_strength=strength)
        new_label = diagnose_trace(out / f"{task['id']}.jsonl", task)
        pairs.append({
            "task_id": lb["task_id"], "config": cfg, "tier": lb["tier"],
            "baseline_success": False,
            "intervened_success": new_label["success"],
            "intervened_at": intervene_at,
            "intervened_first_deviation": new_label["first_deviation_checkpoint"],
        })
        flag = "挽救" if new_label["success"] else "仍失败"
        print(f"[{cfg:9s}] {task['id']} 干预@{intervene_at:12s} → {flag}")
    _write_pairs(pairs)
    return pairs


def run_intervention_real(seed=42, max_retry=3):
    auto_path = ROOT / "data" / "annotations" / "auto_labels.json"
    if not auto_path.exists():
        raise RuntimeError("未找到 auto_labels.json，请先运行 judge_real.py")
    labels = json.loads(auto_path.read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in load_tasks()}

    import os
    import agents
    import judge_real
    from tracer import TrajectoryWriter, build_callback_handler
    model = os.environ.get("AGENT_MODEL", "moonshot-v1-8k")
    llm = agents.get_llm()

    # 增量：已完成的干预配对跳过
    pairs_path = ROOT / "data" / "annotations" / "intervention_pairs.json"
    done = set()
    old_pairs = []
    if pairs_path.exists():
        old_pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
        done = {(p["task_id"], p["config"]) for p in old_pairs}

    pairs = []
    out_dir = ROOT / "data" / "traces_intervened"
    for key, lb in sorted(labels.items()):
        if lb["success"] or lb.get("judge_error"):
            continue  # 只对无干预失败轨迹做干预
        task = tasks[lb["task_id"]]
        cfg = lb["config"]
        if (task["id"], cfg) in done:
            continue
        intervene_at = lb["first_deviation_checkpoint"] or "verification"
        # 无过程偏离但结果错误（罕见）：注入正确结论兜底
        guidance = build_correction(task, intervene_at)
        out = out_dir / cfg
        out.mkdir(parents=True, exist_ok=True)

        err = None
        for attempt in range(1, max_retry + 1):
            writer = TrajectoryWriter(task, cfg, mode="real", seed=seed,
                                      out_dir=out, extra_meta={"model": model,
                                                               "intervention": intervene_at})
            writer.start()
            handler = build_callback_handler(writer)
            try:
                answer = agents.run_real_task(task, cfg, callbacks=[handler],
                                              guidance=guidance)
                err = None
                break
            except Exception as e:
                err = e
                print(f"[{cfg}] {task['id']} 干预重跑第 {attempt} 次失败：{e}")
                if attempt < max_retry:
                    time.sleep(2 ** attempt)
        if err is not None:
            print(f"[{cfg}] {task['id']} 干预重跑放弃：{err}")
            continue
        writer.end(success=None, answer=str(answer)[:800],
                   total_tokens=handler.total_tokens)

        new_label = judge_real.judge_trace(out / f"{task['id']}.jsonl", task, llm=llm)
        rescued = new_label["success"]
        pairs.append({
            "task_id": lb["task_id"], "config": cfg, "tier": lb["tier"],
            "baseline_success": False,
            "intervened_success": rescued,
            "intervened_at": intervene_at,
            "intervened_first_deviation": new_label["first_deviation_checkpoint"],
        })
        flag = "挽救" if rescued else "仍失败"
        print(f"[{cfg:9s}] {task['id']} 干预@{intervene_at:12s} → {flag}")
        _write_pairs(old_pairs + pairs)  # 逐条落盘，中断不丢
    _write_pairs(old_pairs + pairs)
    return old_pairs + pairs


def _write_pairs(pairs):
    out_path = ROOT / "data" / "annotations" / "intervention_pairs.json"
    out_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    n_rescue = sum(1 for p in pairs if p["intervened_success"])
    if pairs:
        print(f"\n干预完成：{len(pairs)} 条失败轨迹，挽救 {n_rescue} 条"
              f"（挽救率 {n_rescue / len(pairs):.1%}）→ {out_path}")
    else:
        print("无失败轨迹需要干预")


def main():
    ap = argparse.ArgumentParser(description="干预实验")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strength", type=float, default=0.25,
                    help="mock 模式：注入纠正后其余检查点偏离概率乘数")
    args = ap.parse_args()
    if args.mode == "mock":
        run_intervention_mock(args.seed, args.strength)
    else:
        run_intervention_real(args.seed)


if __name__ == "__main__":
    main()
