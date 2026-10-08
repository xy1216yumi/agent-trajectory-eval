"""轨迹监控演示工具：输入一条轨迹 JSONL → 自动打检查点标签 → 渲染单文件 HTML 诊断报告。

用法：
  python src/monitor.py data/traces/react/t05.jsonl --out data/reports/demo_report.html
"""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose import diagnose_trace
from mock_agent import load_tasks
from tracer import ROOT, CHECKPOINT_NAMES, read_trace

CSS = """
:root{--ok:#1a7f37;--dev:#cf222e;--bg:#f6f8fa;--card:#fff;--line:#d0d7de}
*{box-sizing:border-box;margin:0}
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:#1f2328;padding:24px}
h1{font-size:20px;margin-bottom:4px}h2{font-size:15px;margin:22px 0 10px}
.meta{color:#57606a;font-size:13px;margin-bottom:14px}
.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:140px}
.card .v{font-size:22px;font-weight:700}.card .k{font-size:12px;color:#57606a}
.ok{color:var(--ok)}.dev{color:var(--dev)}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
th{background:#eef1f4}
.tl{display:flex;flex-direction:column;gap:8px}
.step{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--ok);
  border-radius:6px;padding:10px 14px;cursor:pointer}
.step.deviated{border-left-color:var(--dev);background:#fff5f5}
.step .head{display:flex;gap:10px;align-items:center;font-size:13px;flex-wrap:wrap}
.badge{font-size:11px;padding:1px 8px;border-radius:10px;background:#ddf4e4;color:var(--ok)}
.badge.dev{background:#ffebe9;color:var(--dev)}
.tag{font-size:11px;padding:1px 8px;border-radius:10px;background:#ddf4ff;color:#0969da}
.body{display:none;margin-top:8px;font-size:13px;color:#444;white-space:pre-wrap}
.step.open .body{display:block}
.bar{height:14px;background:#ddf4ff;border-radius:3px;display:inline-block;vertical-align:middle}
.devbar{background:#ffb3b8}
.rule{font-size:12px;color:var(--dev);margin-top:4px}
footer{margin-top:26px;font-size:12px;color:#8b949e}
"""

JS = """
document.querySelectorAll('.step').forEach(s=>s.addEventListener('click',()=>s.classList.toggle('open')));
"""


def render_html(trace_path, task, label, events, result):
    e = html.escape
    first_cp = label["first_deviation_checkpoint"]
    dev_steps = {d["step"]: d for d in label["details"]}
    max_tok = max([s.get("tokens", 0) for s in events if s["event"] == "step"] or [1])

    rows = []
    for cp, name in CHECKPOINT_NAMES.items():
        lb = label["checkpoints"].get(cp, "unknown")
        badge = {"pass": "通过", "deviated": "偏离", "not_reached": "未触达"}.get(lb, "未评判")
        cls = "dev" if lb == "deviated" else "ok"
        reason = (label.get("judge_reasons") or {}).get(cp, "")
        rows.append(f'<tr><td>{name}({cp})</td><td class="{cls}">{badge}</td>'
                    f'<td>{"★ 首次偏离点 " if cp == first_cp else ""}{e(reason)}</td></tr>')

    timeline = []
    for ev in events:
        if ev["event"] == "step":
            d = dev_steps.get(ev["step_index"])
            cls = "step deviated" if d else "step"
            badge = ('<span class="badge dev">偏离</span>' if d
                     else '<span class="badge">正常</span>')
            w = int(200 * ev.get("tokens", 0) / max_tok)
            bar_cls = "bar devbar" if d else "bar"
            rule = f'<div class="rule">判据命中：{e(d["rule"])}</div>' if d else ""
            obs = e(json.dumps(ev.get("observable", {}), ensure_ascii=False, indent=2))
            cp_tag = ev["checkpoint"] or "llm_call"  # real 轨迹的 step 无预设检查点
            timeline.append(
                f'<div class="{cls}"><div class="head">'
                f'<b>#{ev["step_index"]}</b><span class="tag">{e(cp_tag)}</span>'
                f'<span>{e(ev["action"])}</span>{badge}'
                f'<span style="margin-left:auto">{ev.get("tokens",0)} tok / {ev.get("latency_ms",0)} ms</span>'
                f'</div><div style="margin-top:6px"><span class="{bar_cls}" style="width:{w}px"></span></div>'
                f'{rule}<div class="body">{e(ev.get("content",""))}\n\nobservable:\n{obs}</div></div>')
        elif ev["event"] == "tool_call":
            timeline.append(
                f'<div class="step"><div class="head"><b>↳</b><span class="tag">tool_call</span>'
                f'<span>{e(ev["tool"])}({e(json.dumps(ev.get("args",{}),ensure_ascii=False))})</span>'
                f'<span style="margin-left:auto">{ev.get("tokens",0)} tok</span></div>'
                f'<div class="body">{e(json.dumps(ev.get("evidence",[]),ensure_ascii=False,indent=2))}</div></div>')

    success = label["success"]
    verdict = "成功" if success is True else ("失败" if success is False else "已完成（待评判）")
    vcls = "ok" if success is True else "dev"
    waste = label.get("wasted_tokens") or result.get("wasted_tokens_after_deviation", 0)
    total = result.get("total_tokens", 0)
    waste_ratio = (waste / total) if total else 0
    mode = label.get("mode", "mock")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>轨迹诊断报告 - {e(label['task_id'])}</title><style>{CSS}</style></head>
<body>
<h1>Agent 轨迹诊断报告</h1>
<div class="meta">任务 <b>{e(label['task_id'])}</b>（{e(label['tier'])}）｜配置 <b>{e(label['config'])}</b>｜模式 {e(mode)}｜
问题：{e(task['question'])}</div>
<div class="cards">
  <div class="card"><div class="v {vcls}">{verdict}</div><div class="k">最终结论</div></div>
  <div class="card"><div class="v">{label['core_steps']}</div><div class="k">核心步数</div></div>
  <div class="card"><div class="v">{total}</div><div class="k">总 tokens</div></div>
  <div class="card"><div class="v dev">{waste}</div><div class="k">偏离后浪费 tokens（{waste_ratio:.0%}）</div></div>
  <div class="card"><div class="v">{label['first_deviation_step'] or '-'}</div><div class="k">首次偏离步数（{('%.0f%%' % (100*label['first_deviation_step_pct'])) if label['first_deviation_step_pct'] else '-'}）</div></div>
</div>
<h2>检查点标签</h2>
<table><tr><th>检查点</th><th>标签</th><th>备注 / 偏离理由</th></tr>{''.join(rows)}</table>
<h2>轨迹时间线（点击步骤展开详情）</h2>
<div class="tl">{''.join(timeline)}</div>
<footer>由 monitor.py 自动生成 · 判据见 docs/02-偏离判据手册.md · 本文件为自包含单文件报告</footer>
<script>{JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="轨迹诊断 HTML 报告生成器")
    ap.add_argument("trace", help="轨迹 JSONL 路径")
    ap.add_argument("--out", default=None, help="输出 HTML 路径")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    meta, events, result = read_trace(trace_path)
    tasks = {t["id"]: t for t in load_tasks()}
    task = tasks[meta["task_id"]]

    if meta.get("mode") == "real":
        # real 轨迹：规则判据不适用，优先读 judge_real 的标签
        key = f"{meta.get('config', trace_path.parent.name)}/{meta['task_id']}"
        labels_path = ROOT / "data" / "annotations" / "auto_labels.json"
        judged = {}
        if labels_path.exists():
            judged = json.loads(labels_path.read_text(encoding="utf-8")).get(key) or {}
        label = {
            "task_id": meta["task_id"], "config": meta.get("config", ""),
            "tier": meta.get("tier", task["tier"]), "mode": "real",
            "success": judged.get("success", result.get("success")),
            "checkpoints": judged.get("checkpoints", {}),
            "judge_reasons": judged.get("judge_reasons", {}),
            "first_deviation_checkpoint": judged.get("first_deviation_checkpoint"),
            "first_deviation_step": judged.get("first_deviation_step"),
            "first_deviation_step_pct": judged.get("first_deviation_step_pct"),
            "core_steps": judged.get("core_steps") or sum(1 for e in events if e["event"] == "step"),
            "wasted_tokens": judged.get("wasted_tokens", 0),
            "details": judged.get("details", []),
        }
        if not judged:
            print("提示：未找到该轨迹的 judge 标签（auto_labels.json），报告将只渲染原始轨迹")
    else:
        label = diagnose_trace(trace_path, task)
        label["mode"] = "mock"

    out = Path(args.out) if args.out else ROOT / "data" / "reports" / f"{meta['task_id']}_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(trace_path, task, label, events, result), encoding="utf-8")
    verdict = "成功" if label["success"] is True else ("失败" if label["success"] is False else "待评判")
    print(f"诊断报告已生成：{out}")
    print(f"结论={verdict}  "
          f"首次偏离={label['first_deviation_checkpoint'] or '无'}  "
          f"浪费 tokens={label.get('wasted_tokens') or result.get('wasted_tokens_after_deviation', 0)}")


if __name__ == "__main__":
    main()
