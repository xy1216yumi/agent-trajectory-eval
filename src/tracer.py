"""统一轨迹 JSONL schema 与落盘工具。

real 模式：TraceCallbackHandler 基于 LangChain BaseCallbackHandler 记录轨迹；
mock 模式：mock_agent 直接通过 TrajectoryWriter 写同构 JSONL。

事件类型：
  meta      轨迹头：任务/配置/随机种子
  step      一个推理步骤（含检查点标签、可观测产物、token、耗时）
  tool_call 一次工具调用（含工具名、参数、返回的证据条目）
  result    轨迹尾：成功与否、累计成本、首次偏离信息
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS = ["perception", "planning", "tool_use", "verification"]

CHECKPOINT_NAMES = {
    "perception": "感知",
    "planning": "规划",
    "tool_use": "工具调用",
    "verification": "结果校验",
}


class TrajectoryWriter:
    """把一条轨迹写成一个 JSONL 文件（每行一个事件）。"""

    def __init__(self, task, config, mode, seed, out_dir=None, extra_meta=None):
        self.task = task
        self.config = config
        self.mode = mode
        self.seed = seed
        self.extra_meta = extra_meta or {}
        out_dir = Path(out_dir) if out_dir else ROOT / "data" / "traces" / config
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{task['id']}.jsonl"
        self._fh = open(self.path, "w", encoding="utf-8")
        self._t0 = time.time()

    def _write(self, event):
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def start(self):
        self._write({
            "event": "meta",
            "task_id": self.task["id"],
            "tier": self.task["tier"],
            "question": self.task["question"],
            "config": self.config,
            "mode": self.mode,
            "seed": self.seed,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **self.extra_meta,
        })

    def step(self, step_index, checkpoint, action, content, observable=None,
             tokens=0, latency_ms=0):
        self._write({
            "event": "step",
            "step_index": step_index,
            "checkpoint": checkpoint,
            "action": action,
            "content": content,
            "observable": observable or {},
            "tokens": tokens,
            "latency_ms": latency_ms,
        })

    def tool_call(self, step_index, tool, args, expected_tool, evidence=None,
                  tokens=0, latency_ms=0):
        self._write({
            "event": "tool_call",
            "step_index": step_index,
            "tool": tool,
            "args": args,
            "expected_tool": expected_tool,
            "evidence": evidence or [],
            "tokens": tokens,
            "latency_ms": latency_ms,
        })

    def end(self, success, answer, total_tokens, first_deviation=None,
            wasted_tokens=0):
        total_steps = 0
        self._write({
            "event": "result",
            "success": success,
            "answer": answer,
            "total_steps": total_steps,  # 占位，close 时回填
            "total_tokens": total_tokens,
            "elapsed_ms": int((time.time() - self._t0) * 1000),
            "first_deviation": first_deviation,
            "wasted_tokens_after_deviation": wasted_tokens,
            "cost_waste_ratio": round(wasted_tokens / total_tokens, 4) if total_tokens else 0.0,
        })
        self.close()

    def close(self):
        if not self._fh.closed:
            self._fh.close()
        # 回填 total_steps
        lines = self.path.read_text(encoding="utf-8").splitlines()
        n_steps = sum(1 for ln in lines if '"event": "step"' in ln)
        for i, ln in enumerate(lines):
            if '"event": "result"' in ln:
                obj = json.loads(ln)
                obj["total_steps"] = n_steps
                lines[i] = json.dumps(obj, ensure_ascii=False)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_trace(path):
    """读取一条轨迹 JSONL，返回 (meta, events, result)。"""
    events = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = next((e for e in events if e["event"] == "meta"), {})
    result = next((e for e in events if e["event"] == "result"), {})
    return meta, events, result


# ---------------- real 模式：LangChain 回调 ----------------

def build_callback_handler(writer):
    """惰性依赖 langchain：未安装时给出清晰提示。"""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as e:
        raise RuntimeError(
            "real 模式需要 langchain：pip install langchain langchain-openai\n"
            "（mock 模式零依赖，可直接运行）"
        ) from e

    class TraceCallbackHandler(BaseCallbackHandler):
        """把 LLM/工具调用逐步写入 TrajectoryWriter，并累计 token。"""

        def __init__(self):
            super().__init__()
            self._step = 0
            self._t = None
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.reasoning_tokens = 0
            self.cached_tokens = 0

        @staticmethod
        def _extract_usage(response):
            usage = {}
            try:
                usage = (response.llm_output or {}).get("token_usage") or {}
            except Exception:
                pass
            if not usage:
                try:
                    um = response.generations[0][0].message.usage_metadata
                    if um:
                        usage = {
                            "total_tokens": um.get("total_tokens", 0),
                            "prompt_tokens": um.get("input_tokens", 0),
                            "completion_tokens": um.get("output_tokens", 0),
                        }
                        det = (um.get("output_token_details") or {})
                        if det.get("reasoning"):
                            usage["reasoning_tokens"] = det["reasoning"]
                        indet = (um.get("input_token_details") or {})
                        if indet.get("cache_read"):
                            usage["cached_tokens"] = indet["cache_read"]
                except Exception:
                    pass
            return usage

        def on_llm_start(self, serialized, prompts, **kwargs):
            self._step += 1
            self._t = time.time()

        def on_llm_end(self, response, **kwargs):
            usage = self._extract_usage(response)
            tokens = usage.get("total_tokens", 0)
            self.total_tokens += tokens
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.reasoning_tokens += usage.get("reasoning_tokens", 0)
            self.cached_tokens += usage.get("cached_tokens", 0)
            text, reasoning = "", ""
            try:
                msg = response.generations[0][0].message
                text = (response.generations[0][0].text or "")[:800]
                reasoning = str(msg.additional_kwargs.get("reasoning_content", ""))[:1200]
            except Exception:
                pass
            writer.step(self._step, None, "llm_call", text,
                        observable={"usage": usage, "reasoning": reasoning},
                        tokens=tokens,
                        latency_ms=int((time.time() - self._t) * 1000) if self._t else 0)

        def on_tool_start(self, serialized, input_str, **kwargs):
            self._t = time.time()
            self._tool_name = serialized.get("name", "unknown")
            self._tool_input = input_str

        def on_tool_end(self, output, **kwargs):
            writer.tool_call(self._step, getattr(self, "_tool_name", "unknown"),
                             {"input": getattr(self, "_tool_input", "")[:300]},
                             expected_tool=None,
                             evidence=[{"id": "RAW", "fact": str(output)[:500]}],
                             tokens=0,
                             latency_ms=int((time.time() - self._t) * 1000) if self._t else 0)

    return TraceCallbackHandler()
