"""real 模式：三组配置的真实 LangChain 实现（惰性导入）。

  build_direct     直调：一次 LLM 调用直接出答案
  build_react      ReAct：带 search / citation_check 工具的 ReAct Agent
  build_reflexion  反思：ReAct 输出后再做一轮自我批评与修正

LLM 走 OpenAI 兼容接口，默认指向 Moonshot/Kimi：
  export MOONSHOT_API_KEY=sk-...        # 或 OPENAI_API_KEY
  export MOONSHOT_BASE_URL=...          # 可选，默认 https://api.moonshot.cn/v1
  export AGENT_MODEL=moonshot-v1-8k     # 可选
"""
import os


def get_llm():
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "real 模式需要可选依赖：pip install langchain langchain-openai langgraph"
        ) from e
    api_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 API key。请设置环境变量 MOONSHOT_API_KEY（或 OPENAI_API_KEY）后重试；\n"
            "没有 key 可先用 mock 模式：python src/run_experiment.py --mode mock"
        )
    return ChatOpenAI(
        model=os.environ.get("AGENT_MODEL", "moonshot-v1-8k"),
        base_url=os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        api_key=api_key,
        # kimi-k3 为 thinking 模型，仅允许 temperature=1；其他模型可用 AGENT_TEMPERATURE 覆盖
        temperature=float(os.environ.get("AGENT_TEMPERATURE", "1")),
        timeout=120,
        max_retries=2,
        # 本环境 openai 3.16 自带的 httpx2 与 httpx 0.28 传输层不兼容，显式注入 httpx 客户端绕过
        http_client=httpx.Client(timeout=120),
    )


def _make_tools(task):
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        """检索科研来源，返回与 query 相关的证据条目。"""
        facts = task.get("evidence_facts") or [f"来源记录：{task['ground_truth']}"]
        return "\n".join(f"EV-{i+1}: {f}" for i, f in enumerate(facts))

    @tool
    def citation_check(citation: str) -> str:
        """核验一条引用是否真实存在于已检索来源中。"""
        facts = task.get("evidence_facts") or [task["ground_truth"]]
        ok = any(citation.strip() and citation.strip() in f for f in facts)
        return "引用有效" if ok else "引用不存在或与证据不符"

    return [search, citation_check]


def build_direct(task, llm=None):
    """直调：单个 prompt 直接要结论，无规划、无校验。"""
    llm = llm or get_llm()
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是科研信息核对助手。请直接回答用户的研究问题。"),
        ("user", "{question}"),
    ])
    chain = prompt | llm

    def run(question, callbacks=None):
        return chain.invoke({"question": question},
                            config={"callbacks": callbacks}).content

    return run


def build_react(task, llm=None):
    """ReAct：推理-行动交替，可使用检索与引用核验工具。"""
    llm = llm or get_llm()
    try:
        from langgraph.prebuilt import create_react_agent  # langchain 1.x
    except ImportError:
        try:
            from langchain.agents import create_react_agent  # langchain 0.x 经典路径
        except ImportError as e:
            raise RuntimeError(
                "未找到 create_react_agent，请安装 langgraph（langchain 1.x）或 langchain<1.0"
            ) from e
    agent = create_react_agent(llm, _make_tools(task))

    def run(question, callbacks=None):
        out = agent.invoke(
            {"messages": [("user", f"你是科研信息核对 Agent。回答前必须检索并交叉核对，"
                                   f"最后给出含引用的结构化结论。\n问题：{question}")]},
            config={"callbacks": callbacks, "recursion_limit": 20})
        return out["messages"][-1].content

    return run


def build_reflexion(task, llm=None):
    """反思：ReAct 结论后再做一轮自我批评，发现问题则修正重答。"""
    llm = llm or get_llm()
    from langchain_core.prompts import ChatPromptTemplate
    base = build_react(task, llm)
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是严格的审稿人。检查结论是否：①覆盖题目全部关键条件；"
                   "②每条引用真实存在；③边界情形（信息冲突/来源缺失）是否如实说明。"
                   "若有任何问题，逐条指出；若无问题，回复“通过”。"),
        ("user", "问题：{question}\n\n待审结论：{draft}"),
    ])
    critic = critic_prompt | llm

    def run(question, callbacks=None):
        draft = base(question, callbacks=callbacks)
        critique = critic.invoke({"question": question, "draft": draft},
                                 config={"callbacks": callbacks}).content
        if "通过" in critique:
            return draft
        fix_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据审稿意见修正结论，输出最终答案。"),
            ("user", "问题：{question}\n原结论：{draft}\n审稿意见：{critique}"),
        ])
        return (fix_prompt | llm).invoke(
            {"question": question, "draft": draft, "critique": critique},
            config={"callbacks": callbacks}).content

    return run


BUILDERS = {"direct": build_direct, "react": build_react, "reflexion": build_reflexion}


def run_real_task(task, config, callbacks=None, guidance=None):
    """跑一条真实轨迹，返回结论文本。guidance：干预实验注入的检查点确认信息。"""
    runner = BUILDERS[config](task)
    question = task["question"]
    if guidance:
        question = (f"【检查点确认】人工复核已确认以下正确信息，请严格据此继续完成任务：\n"
                    f"{guidance}\n\n原始问题：{question}")
    return runner(question, callbacks=callbacks)
