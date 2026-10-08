"""脚本化模拟 Agent（mock 模式，零依赖、可复现）。

三组配置按每检查点偏离概率植入偏离行为：
  direct    直调：无规划无校验，偏离最多
  react     ReAct：带检索/引用核验工具，次之
  reflexion 带反思循环，最少

token 模型假设：prompt 随历史累积；偏离发生后 Agent 进入"困惑重试"，
每次重试需重放已被污染的完整上下文——因此偏离越晚，事后消耗 token 越多。
"""
import random
from pathlib import Path

from tracer import TrajectoryWriter, ROOT

GROUP_CONFIGS = {
    "direct": {
        "desc": "直调：一次性生成答案，无规划与校验",
        "dev_prob": {"perception": 0.12, "planning": 0.18, "tool_use": 0.27, "verification": 0.40},
    },
    "react": {
        "desc": "ReAct：推理-行动交替，带检索与引用核验",
        "dev_prob": {"perception": 0.08, "planning": 0.12, "tool_use": 0.18, "verification": 0.25},
    },
    "reflexion": {
        "desc": "反思：每步自检后再前进",
        "dev_prob": {"perception": 0.03, "planning": 0.05, "tool_use": 0.08, "verification": 0.12},
    },
}

TIER_MULTIPLIER = {"easy": 0.7, "medium": 1.0, "hard": 1.4, "edge": 1.0}
EDGE_VERIFICATION_BOOST = 1.6  # 边界题在校验环节更容易失守（应拒绝/应报冲突而未做）

BASE_PROMPT = 180          # 系统提示等固定开销
CONTENT_RANGE = {          # 每步输出 token 区间（随环节递增）
    "perception": (160, 220),
    "planning": (200, 260),
    "tool_use": (260, 340),
    "verification": (360, 440),
}
LATENCY_RANGE = {"perception": (200, 400), "planning": (250, 450),
                 "tool_use": (500, 900), "verification": (400, 800)}

# 供模拟检索用的证据库（按任务 id 缺省生成）
DEFAULT_FACTS = ["来源A记录：{gt}", "来源B交叉确认：{gt}"]


def load_tasks(path=None):
    import json
    path = Path(path) if path else ROOT / "tasks" / "tasks.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))["tasks"]


def _dev_prob(config, checkpoint, tier):
    p = GROUP_CONFIGS[config]["dev_prob"][checkpoint] * TIER_MULTIPLIER[tier]
    if tier == "edge" and checkpoint == "verification":
        p = GROUP_CONFIGS[config]["dev_prob"][checkpoint] * EDGE_VERIFICATION_BOOST
    return min(p, 0.9)


def _make_evidence(task, rng):
    facts = task.get("evidence_facts") or [f.format(gt=task["ground_truth"]) for f in DEFAULT_FACTS]
    return [{"id": f"EV-{i+1}", "fact": f} for i, f in enumerate(facts)]


def run_mock_task(task, config, seed, out_dir=None, intervene_at=None,
                  intervention_strength=0.25):
    """跑一条模拟轨迹并落盘。

    intervene_at: 若干检查点被注入纠正信息，该检查点偏离概率清零，
                  其余检查点偏离概率乘以 intervention_strength。
    返回结果 dict（含 success / 首次偏离等）。
    """
    rng = random.Random(f"{seed}|{task['id']}|{config}|{intervene_at}")
    tier = task["tier"]
    is_edge = tier == "edge"
    expected_behavior = task.get("expected_behavior", "answer")
    evidence = _make_evidence(task, rng)
    steps_spec = task["required_steps"]

    def prob_of(cp):
        if intervene_at == cp:
            return 0.0
        p = _dev_prob(config, cp, tier)
        return p * intervention_strength if intervene_at else p

    # 每个检查点只掷一次骰（在该检查点的首个步骤生效），避免多步任务概率叠加
    dev_roll = {cp: rng.random() < prob_of(cp) for cp in
                ["perception", "planning", "tool_use", "verification"]}
    rolled = set()

    writer = TrajectoryWriter(task, config, mode="mock", seed=seed, out_dir=out_dir)
    writer.start()

    first_dev = None
    total_tokens = 0
    tokens_after_dev = 0
    cum_content = 0
    step_index = 0
    deviated = set()

    def bill(cp):
        nonlocal total_tokens, tokens_after_dev, cum_content
        lo, hi = CONTENT_RANGE[cp]
        content_tok = rng.randint(lo, hi)
        prompt_tok = BASE_PROMPT + cum_content
        tok = prompt_tok + content_tok
        cum_content += content_tok
        total_tokens += tok
        if first_dev:
            tokens_after_dev += tok
        lat = rng.randint(*LATENCY_RANGE[cp])
        return tok, lat

    def mark_dev(cp, dtype, detail):
        nonlocal first_dev
        deviated.add(cp)
        if first_dev is None:
            first_dev = {"checkpoint": cp, "step": step_index, "type": dtype, "detail": detail}

    retrieved_ids = set()

    for spec in steps_spec:
        cp = spec["checkpoint"]
        name = spec["name"]
        step_index += 1
        deviate = dev_roll[cp] and cp not in rolled  # 每检查点只在首个步骤生效
        rolled.add(cp)

        if cp == "perception":
            tok, lat = bill(cp)
            conds = list(task["key_conditions"])
            if deviate:
                if rng.random() < 0.5 and len(conds) > 1:
                    dropped = conds.pop()  # 漏读
                    mark_dev(cp, "漏读条件", f"漏掉关键条件：{dropped}")
                    content = f"解析题目，提取到 {len(conds)} 条关键条件（实际漏掉 1 条）。"
                else:
                    wrong = f"[误读]{conds[0]}（方向/数值被曲解）"
                    mark_dev(cp, "误读条件", f"将「{conds[0]}」曲解为其反向/错误口径")
                    conds[0] = wrong
                    content = "解析题目，提取关键条件，但其中一条被误读。"
            else:
                content = f"解析题目，完整提取 {len(conds)} 条关键条件。"
            writer.step(step_index, cp, name, content,
                        observable={"conditions_extracted": conds}, tokens=tok, latency_ms=lat)

        elif cp == "planning":
            tok, lat = bill(cp)
            plan = [s["name"] for s in steps_spec]
            if deviate:
                if rng.random() < 0.5 and len(plan) > 3:
                    skipped = plan.pop(len(plan) // 2)  # 跳步
                    mark_dev(cp, "步骤缺失", f"计划中缺少必要步骤：{skipped}")
                    content = f"制定核对计划（缺少步骤：{skipped}）。"
                else:
                    i = rng.randrange(len(plan) - 1)
                    plan[i], plan[i + 1] = plan[i + 1], plan[i]  # 顺序错误
                    mark_dev(cp, "顺序错误", f"步骤「{plan[i+1]}」与「{plan[i]}」顺序颠倒")
                    content = "制定核对计划，但步骤顺序有误。"
            else:
                content = "制定核对计划：明确条件→检索→交叉核对→结论。"
            writer.step(step_index, cp, name, content,
                        observable={"plan": plan}, tokens=tok, latency_ms=lat)

        elif cp == "tool_use":
            tok, lat = bill(cp)
            tool_tok, tool_lat = rng.randint(80, 150), rng.randint(100, 400)
            total_tokens += tool_tok
            if first_dev:
                tokens_after_dev += tool_tok
            if deviate:
                kind = rng.choice(["wrong_tool", "bad_args", "missing"])
                if kind == "wrong_tool":
                    mark_dev(cp, "调用错误工具", "应调用 search，实际调用 calculator")
                    writer.tool_call(step_index, "calculator", {"expression": "1+1"},
                                     expected_tool="search", evidence=[],
                                     tokens=tool_tok, latency_ms=tool_lat)
                    content = "尝试获取证据，但调用了错误的工具。"
                elif kind == "bad_args":
                    mark_dev(cp, "参数错误", "检索 query 与题目无关")
                    writer.tool_call(step_index, "search", {"query": "无关关键词"},
                                     expected_tool="search", evidence=[],
                                     tokens=tool_tok, latency_ms=tool_lat)
                    content = "发起检索，但查询词与问题不匹配，未命中有效来源。"
                else:  # 该用未用：不发 tool_call，凭记忆断言
                    mark_dev(cp, "该用未用", "需要检索核实，但直接凭记忆作答")
                    content = "（未调用检索工具，直接凭印象继续）"
            else:
                ev = evidence[:2] if len(retrieved_ids) == 0 else evidence
                for e in ev:
                    retrieved_ids.add(e["id"])
                writer.tool_call(step_index, "search", {"query": task["question"][:60]},
                                 expected_tool="search", evidence=ev,
                                 tokens=tool_tok, latency_ms=tool_lat)
                content = f"检索并记录 {len(ev)} 条证据：{', '.join(e['id'] for e in ev)}。"
            writer.step(step_index, cp, name, content,
                        observable={"retrieved_ids": sorted(retrieved_ids)},
                        tokens=tok, latency_ms=lat)

        elif cp == "verification":
            tok, lat = bill(cp)
            final_behavior = expected_behavior
            citations = [{"source_id": e["id"], "claim": e["fact"]} for e in evidence if e["id"] in retrieved_ids]
            if not citations:  # 之前工具环节已偏离，无可用证据
                citations = [{"source_id": "EV-99", "claim": f"凭记忆：{task['ground_truth']}"}]
            if deviate:
                kind = "未按预期行为作答" if is_edge else rng.choice(["编造引用", "结论与证据不符"])
                if kind == "编造引用":
                    citations = citations + [{"source_id": "EV-99", "claim": "不存在的来源支撑结论"}]
                    mark_dev(cp, kind, "引用了检索结果中不存在的来源 EV-99")
                elif kind == "结论与证据不符":
                    citations = [{"source_id": c["source_id"],
                                  "claim": "与所引证据内容相反的结论表述"}
                                 for c in citations]
                    mark_dev(cp, kind, "结论表述与所引证据内容相反")
                else:
                    final_behavior = "answer"
                    mark_dev(cp, kind, f"边界题应 {expected_behavior}，实际直接给出确定答案")
                content = "输出结论并完成校验（存在偏离）。"
            else:
                content = "校验引用与证据一致，输出结构化结论。"
            writer.step(step_index, cp, name, content,
                        observable={"citations": citations, "final_behavior": final_behavior,
                                    "expected_behavior": expected_behavior},
                        tokens=tok, latency_ms=lat)
        else:
            raise ValueError(f"未知检查点：{cp}")

    success = not first_dev  # 边界题：校验环节未偏离即已按预期行为作答

    # 偏离后的"困惑重试"：重试轮数 = 已走步数；每轮输出又追加进上下文，
    # 下一轮重试需为更长的污染上下文付费——重试成本随偏离步数近似平方增长
    if first_dev:
        recheck_ctx = 0
        for j in range(first_dev["step"]):
            step_index += 1
            content_tok = rng.randint(400, 600)
            prompt_tok = BASE_PROMPT + cum_content + recheck_ctx
            tok = prompt_tok + content_tok
            recheck_ctx += content_tok
            tok = prompt_tok + content_tok
            total_tokens += tok
            tokens_after_dev += tok
            writer.step(step_index, first_dev["checkpoint"], "纠偏重试",
                        f"带着已有上下文自我确认第 {j+1} 次，仍未发现偏离。",
                        observable={}, tokens=tok, latency_ms=rng.randint(200, 500))

    if first_dev:
        answer = "（错误结论）" if not is_edge else "（未按预期行为作答）"
    else:
        answer = task["ground_truth"] if not is_edge else f"已按预期行为处理：{expected_behavior}"

    writer.end(success=success, answer=answer, total_tokens=total_tokens,
               first_deviation=first_dev, wasted_tokens=tokens_after_dev)

    return {
        "task_id": task["id"], "config": config, "success": success,
        "first_deviation": first_dev, "total_tokens": total_tokens,
        "wasted_tokens": tokens_after_dev, "path": str(writer.path),
    }
