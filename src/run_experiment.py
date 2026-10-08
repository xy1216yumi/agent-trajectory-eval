"""跑 30 任务 × 3 组配置，输出轨迹到 data/traces/{config}/*.jsonl。

用法：
  python src/run_experiment.py --mode mock --seed 42
  python src/run_experiment.py --mode real [--configs react] [--tasks t01 t02]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_agent import GROUP_CONFIGS, load_tasks, run_mock_task
from tracer import ROOT


def run_mock(tasks, configs, seed):
    results = []
    for cfg in configs:
        for task in tasks:
            r = run_mock_task(task, cfg, seed)
            results.append(r)
            flag = "OK " if r["success"] else "DEV"
            dev = r["first_deviation"]["checkpoint"] if r["first_deviation"] else "-"
            print(f"[{cfg:9s}] {task['id']} {flag} 首次偏离={dev:12s} tokens={r['total_tokens']}")
    return results


def run_real(tasks, configs, seed, max_retry=3):
    import time as _time
    from tracer import TrajectoryWriter, build_callback_handler
    import agents  # 惰性导入 langchain
    model = os.environ.get("AGENT_MODEL", "moonshot-v1-8k")
    results = []
    for cfg in configs:
        for task in tasks:
            err, answer, handler, writer = None, None, None, None
            for attempt in range(1, max_retry + 1):
                writer = TrajectoryWriter(task, cfg, mode="real", seed=seed,
                                          extra_meta={"model": model})
                writer.start()
                handler = build_callback_handler(writer)
                try:
                    answer = agents.run_real_task(task, cfg, callbacks=[handler])
                    err = None
                    break
                except Exception as e:  # 单任务失败不中断整批，指数退避重试
                    err = e
                    print(f"[{cfg}] {task['id']} 第 {attempt} 次调用失败：{e}")
                    if attempt < max_retry:
                        _time.sleep(2 ** attempt)
            total = handler.total_tokens if handler else 0
            if err is None:
                # real 模式的 success 不作硬判定：completed=True 表示"跑完"，
                # 真正的成功判定交给 judge_real.py / 人工标注
                writer.end(success=None, answer=str(answer)[:800], total_tokens=total,
                           first_deviation=None, wasted_tokens=0)
                print(f"[{cfg}] {task['id']} 完成  tokens={total}（成功判定见 judge_real）")
                results.append({"task_id": task["id"], "config": cfg, "success": None,
                                "completed": True, "total_tokens": total})
            else:
                writer.end(success=None, answer=f"运行失败：{err}", total_tokens=total,
                           first_deviation=None, wasted_tokens=0)
                results.append({"task_id": task["id"], "config": cfg, "success": None,
                                "completed": False, "total_tokens": total})
    return results


def main():
    ap = argparse.ArgumentParser(description="采集 30 任务 × 3 组配置的 Agent 轨迹")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="*", default=list(GROUP_CONFIGS))
    ap.add_argument("--tasks", nargs="*", default=None, help="只跑指定任务 id")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in args.tasks]

    print(f"模式={args.mode}  任务数={len(tasks)}  配置={args.configs}  seed={args.seed}")
    if args.mode == "mock":
        results = run_mock(tasks, args.configs, args.seed)
    else:
        results = run_real(tasks, args.configs, args.seed)

    # 成本汇总
    print("\n===== 成本汇总 =====")
    for cfg in args.configs:
        rs = [r for r in results if r["config"] == cfg]
        toks = sum(r.get("total_tokens", 0) for r in rs)
        if args.mode == "real":
            n_done = sum(1 for r in rs if r.get("completed"))
            print(f"{cfg:10s} 完成 {n_done}/{len(rs)}  总 tokens={toks}  平均 tokens={toks // max(len(rs), 1)}")
        else:
            n_ok = sum(1 for r in rs if r["success"])
            print(f"{cfg:10s} 成功 {n_ok}/{len(rs)}  总 tokens={toks}  平均 tokens={toks // max(len(rs), 1)}")
    print(f"轨迹输出目录：{ROOT / 'data' / 'traces'}")


if __name__ == "__main__":
    main()
