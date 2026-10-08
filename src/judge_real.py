"""LLM-as-judge：对 real 轨迹逐检查点标注（默认 kimi-k3，temperature=0）。

用法（需先桥接环境变量，见 README）：
  D:/Anaconda_envs/envs/langchain_env/python.exe src/judge_real.py [--workers 4]

流程：把每条轨迹渲染成可读 transcript，连同任务字段与精简版判据手册发给 LLM，
要求输出结构化 JSON；随后用代码确定性计算 first_deviation_step_pct、
wasted_tokens、cost_waste_ratio，输出与 analyze.py 兼容的 auto_labels.json。
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_agent import load_tasks
from tracer import ROOT, CHECKPOINTS, read_trace

CRITERIA_BRIEF = """【偏离判据手册（精简版）】
标签：pass=通过 / deviated=偏离 / not_reached=该检查点未执行到。
1. 感知 perception（对照 key_conditions）：
   - 漏读：任一关键条件未被理解/未体现 → deviated
   - 误读：条件被曲解（年份/数值/比较方向/阈值/口径错误） → deviated
2. 规划 planning（对照 required_steps）：
   - 步骤缺失：必要步骤（如交叉核对）完全未执行 → deviated
   - 顺序错误：环节顺序颠倒（如下结论先于检索） → deviated
3. 工具调用 tool_use：
   - 调用错误工具 / 检索 query 与问题明显不匹配 → deviated
   - 该用未用：任务要求检索核实，却完全凭记忆作答（全程无 search 调用） → deviated
4. 结果校验 verification：
   - 编造引用：引用的来源编号/内容未出现在工具返回中 → deviated
   - 结论与证据不符：结论方向与检索到的证据相反 → deviated
   - 应拒绝/应说明冲突而未做（edge 题，对照 expected_behavior） → deviated
判定原则：只看行为不看说法；配置设计不豁免（任务 required_steps 明确要求而未做即算偏离，
三组配置的架构差异正是评测对象）；级联错误每个检查点独立判定。"""

JUDGE_PROMPT = """你是严格的 Agent 轨迹标注员。请根据判据手册，对下面这条"科研信息核对 Agent"的执行轨迹逐检查点标注。

{criteria}

【任务信息】
- 难度 tier：{tier}
- 问题：{question}
- 标准答案 ground_truth：{ground_truth}
- 关键条件 key_conditions：{key_conditions}
- 标准流程 required_steps：{required_steps}
- edge 期望行为 expected_behavior：{expected_behavior}
- 检索工具可返回的证据 evidence_facts：{evidence_facts}

【轨迹 transcript】（Agent 配置：{config}）
{transcript}

【输出要求】只输出一个 JSON 对象（不要输出其他任何文字），结构：
{{
  "perception": {{"label": "pass|deviated|not_reached", "reason": "一句话理由"}},
  "planning": {{"label": "...", "reason": "..."}},
  "tool_use": {{"label": "...", "reason": "..."}},
  "verification": {{"label": "...", "reason": "..."}},
  "first_deviation": {{"step_index": 整数或 null, "checkpoint": "perception|planning|tool_use|verification 或 null"}},
  "answer_correct": true 或 false,
  "final_behavior": "answer|refuse|report_conflict",
  "final_answer_summary": "Agent 最终结论的一句话概括"
}}
注意：first_deviation 是所有 deviated 检查点中 step_index 最小者；无偏离时两个字段都为 null；
answer_correct 只对照 ground_truth 判断最终结论对错（edge 题对照 expected_behavior 判断 final_behavior 是否达标，
answer_correct 字段同样填写最终结论事实性是否正确）。"""


def render_transcript(events):
    lines = []
    for e in events:
        if e["event"] == "step":
            reasoning = (e.get("observable") or {}).get("reasoning", "")
            lines.append(f"[步骤 {e['step_index']}] LLM 调用")
            if reasoning:
                lines.append(f"  推理：{reasoning}")
            if e.get("content"):
                lines.append(f"  输出：{e['content']}")
        elif e["event"] == "tool_call":
            ev_text = " | ".join(f"{x['id']}: {x['fact'][:200]}" for x in e.get("evidence", []))
            lines.append(f"[步骤 {e['step_index']}] 工具调用 {e['tool']} 参数={e.get('args')}")
            lines.append(f"  返回：{ev_text[:600]}")
        elif e["event"] == "result":
            lines.append(f"[结束] 最终答案：{e.get('answer', '')[:500]}")
    return "\n".join(lines)


def _parse_judge_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("judge 输出中未找到 JSON")
    obj = json.loads(m.group(0))
    for cp in CHECKPOINTS:
        assert obj[cp]["label"] in ("pass", "deviated", "not_reached"), f"{cp} 标签非法"
    fd = obj.get("first_deviation") or {}
    deviated = [cp for cp in CHECKPOINTS if obj[cp]["label"] == "deviated"]
    if deviated:
        assert fd.get("checkpoint") in deviated, "首次偏离检查点与逐点标签不一致"
        assert isinstance(fd.get("step_index"), int), "首次偏离缺少 step_index"
    else:
        fd = {"step_index": None, "checkpoint": None}
    assert isinstance(obj.get("answer_correct"), bool)
    assert obj.get("final_behavior") in ("answer", "refuse", "report_conflict")
    return obj, fd


def judge_trace(path, task, llm=None, max_retry=3):
    """评判一条 real 轨迹，返回与 analyze.py 兼容的标签 dict。"""
    import agents
    llm = llm or agents.get_llm()
    meta, events, result = read_trace(path)
    transcript = render_transcript(events)
    prompt = JUDGE_PROMPT.format(
        criteria=CRITERIA_BRIEF,
        tier=task["tier"], question=task["question"],
        ground_truth=task["ground_truth"],
        key_conditions=json.dumps(task["key_conditions"], ensure_ascii=False),
        required_steps=json.dumps([s["name"] for s in task["required_steps"]], ensure_ascii=False),
        expected_behavior=task.get("expected_behavior", "answer"),
        evidence_facts=json.dumps(task.get("evidence_facts", []), ensure_ascii=False),
        config=meta.get("config", ""), transcript=transcript)

    obj, fd, raw = None, None, ""
    for attempt in range(1, max_retry + 1):
        try:
            resp = llm.invoke(prompt)
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            obj, fd = _parse_judge_json(raw)
            break
        except Exception as e:
            print(f"  judge {path.name} 第 {attempt} 次失败：{e}")
            if attempt < max_retry:
                time.sleep(2 ** attempt)
    if obj is None:
        # 解析最终失败：打标记，人工复核
        obj = {cp: {"label": "not_reached", "reason": "judge 解析失败"} for cp in CHECKPOINTS}
        fd = {"step_index": None, "checkpoint": None}
        obj.update({"answer_correct": False, "final_behavior": "answer",
                    "final_answer_summary": "", "judge_error": True})

    steps = [e for e in events if e["event"] == "step"]
    n_steps = len(steps)
    total_tokens = result.get("total_tokens", 0)
    first_step = fd["step_index"]
    wasted = 0
    if first_step:
        wasted = sum(e.get("tokens", 0) for e in events
                     if e["event"] in ("step", "tool_call")
                     and e.get("step_index", 0) >= first_step)
    is_edge = task["tier"] == "edge"
    success = (obj["final_behavior"] == task.get("expected_behavior")) if is_edge \
        else obj["answer_correct"]

    return {
        "task_id": meta.get("task_id", task["id"]),
        "config": meta.get("config", ""),
        "tier": meta.get("tier", task["tier"]),
        "success": bool(success),
        "answer_correct": obj["answer_correct"],
        "final_behavior": obj["final_behavior"],
        "final_answer_summary": obj.get("final_answer_summary", ""),
        "checkpoints": {cp: obj[cp]["label"] for cp in CHECKPOINTS},
        "judge_reasons": {cp: obj[cp]["reason"] for cp in CHECKPOINTS},
        "first_deviation_checkpoint": fd["checkpoint"],
        "first_deviation_step": first_step,
        "first_deviation_step_pct": round(first_step / n_steps, 4) if first_step and n_steps else None,
        "core_steps": n_steps,
        "total_tokens": total_tokens,
        "wasted_tokens": wasted,
        "cost_waste_ratio": round(wasted / total_tokens, 4) if total_tokens else 0.0,
        "details": [{"checkpoint": fd["checkpoint"], "step": first_step,
                     "rule": obj[fd["checkpoint"]]["reason"]}] if fd["checkpoint"] else [],
        **({"judge_error": True} if obj.get("judge_error") else {}),
    }


def main():
    ap = argparse.ArgumentParser(description="real 轨迹 LLM 评判")
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="只评判指定 key，如 react/t05")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir) if args.traces_dir else ROOT / "data" / "traces"
    out_path = Path(args.out) if args.out else ROOT / "data" / "annotations" / "auto_labels.json"
    tasks = {t["id"]: t for t in load_tasks()}
    files = sorted(traces_dir.glob("**/*.jsonl"))
    if args.only:
        want = set(args.only)
        files = [f for f in files if f"{f.parent.name}/{f.stem}" in want]

    import agents
    llm = agents.get_llm()
    labels = {}
    if out_path.exists():  # 增量：已有结果跳过，避免重复调用
        labels = json.loads(out_path.read_text(encoding="utf-8"))
    todo = []
    for f in files:
        meta, _, _ = read_trace(f)
        key = f"{meta.get('config', f.parent.name)}/{meta['task_id']}"
        if key not in labels and meta["task_id"] in tasks:
            todo.append((key, f, tasks[meta["task_id"]]))
    print(f"待评判 {len(todo)} 条（已有 {len(labels)} 条），workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_trace, f, t, llm): key for key, f, t in todo}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                labels[key] = fut.result()
                lb = labels[key]
                flag = "OK " if lb["success"] else "DEV"
                print(f"[{flag}] {key} 首次偏离={lb['first_deviation_checkpoint']} "
                      f"步={lb['first_deviation_step']} tokens={lb['total_tokens']}"
                      + (" [judge_error]" if lb.get("judge_error") else ""))
            except Exception as e:
                print(f"[ERROR] {key}: {e}")
            out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    n_err = sum(1 for lb in labels.values() if lb.get("judge_error"))
    print(f"评判完成：{len(labels)} 条 → {out_path}（judge_error {n_err} 条）")


if __name__ == "__main__":
    main()
