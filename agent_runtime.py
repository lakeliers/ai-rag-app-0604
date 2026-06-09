import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import rag_agent_core as agent


ENABLE_LLM_PLANNER = os.getenv("ENABLE_LLM_PLANNER", "1") == "1"
PLANNER_MODEL = os.getenv("PLANNER_MODEL", agent.DEEPSEEK_MODEL)


@dataclass
class AgentStep:
    name: str
    tool: str
    reason: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    status: str
    summary: str
    data: Any = None
    elapsed_ms: int = 0
    error: str = ""


def tool_web_collect(question: str, max_results: int) -> ToolResult:
    ingested = agent.web_collect(question, max_results=max_results)
    return ToolResult(
        status="success",
        summary=f"联网收集完成，写入 {len(ingested)} 条网页资料。",
        data=ingested,
    )


def tool_rag_search(question: str, top_k: int, preferred_sources: list[str]) -> ToolResult:
    results = agent.search_chroma(
        question,
        top_k=top_k,
        preferred_sources=preferred_sources,
    )
    upload_count = sum(1 for item in results if item.get("source_type") == "upload")
    web_count = sum(1 for item in results if item.get("source_type") == "web")
    return ToolResult(
        status="success",
        summary=f"检索完成，选出 {len(results)} 条资料，其中上传资料 {upload_count} 条、网页资料 {web_count} 条。",
        data=results,
    )


def tool_generate_answer(question: str, search_results: list[dict[str, Any]]) -> ToolResult:
    answer = agent.ask_deepseek(question, search_results)
    if answer is None:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY。")

    agent.conversation_history.append({"role": "user", "content": question})
    agent.conversation_history.append({"role": "assistant", "content": answer})

    return ToolResult(
        status="success",
        summary="回答生成完成。",
        data=answer,
    )


TOOLS: dict[str, Callable[..., ToolResult]] = {
    "web_collect": tool_web_collect,
    "rag_search": tool_rag_search,
    "generate_answer": tool_generate_answer,
}


PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_collect",
            "description": "联网搜索并收集与用户问题相关的公开网页资料，适合最新信息、外部事实、行业趋势、没有上传资料时的补充检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用于网页搜索的查询词，通常使用用户原问题或稍微改写后的搜索词。",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多收集的网页结果数。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要联网收集资料。",
                    },
                },
                "required": ["question", "max_results", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "在本地知识库中检索、融合排序、rerank、去重并选择要送给大模型的资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户问题。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "最终选择的资料条数。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要做 RAG 检索。",
                    },
                },
                "required": ["question", "top_k", "reason"],
            },
        },
    },
]


def build_rule_based_steps(
    question: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str] | None = None,
) -> list[AgentStep]:
    preferred_sources = preferred_sources or []
    steps: list[AgentStep] = []

    if use_web:
        if preferred_sources:
            reason = "用户上传资料优先，但仍联网补充外部资料。"
        else:
            reason = "没有用户上传资料时，先联网收集资料再检索。"
        steps.append(
            AgentStep(
                name="联网收集资料",
                tool="web_collect",
                reason=reason,
                args={"question": question, "max_results": web_max_results},
            )
        )

    steps.append(
        AgentStep(
            name="RAG 检索排序",
            tool="rag_search",
            reason="在知识库中做混合召回、重排、去重和上下文打包。",
            args={
                "question": question,
                "top_k": top_k,
                "preferred_sources": preferred_sources,
            },
        )
    )
    steps.append(
        AgentStep(
            name="生成最终回答",
            tool="generate_answer",
            reason="把筛选后的资料交给大模型生成带依据的回答。",
            args={"question": question, "search_results": "$rag_search"},
        )
    )

    return steps


def build_planner_prompt(
    question: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str],
) -> str:
    upload_state = "有用户上传资料" if preferred_sources else "没有用户上传资料"
    web_state = "允许联网" if use_web else "不允许联网"
    return f"""你是一个 RAG Agent 的工具规划器，只负责决定接下来要调用哪些工具。

当前状态：
- {upload_state}
- {web_state}
- 资料条数 top_k={top_k}
- 网页结果数 web_max_results={web_max_results}

规划规则：
1. 你只能通过工具调用表达计划，不要输出自然语言答案。
2. rag_search 通常必须调用，因为最终回答需要证据资料。
3. 如果没有用户上传资料且允许联网，必须先调用 web_collect，再调用 rag_search。
4. 如果问题涉及最新趋势、新闻、当前状态、外部事实，即使有上传资料，也优先调用 web_collect 补充资料。
5. 如果问题明显只要求总结用户上传资料，可以跳过 web_collect，直接 rag_search。
6. 不要调用最终回答工具，最终回答由系统在检索后统一生成。

用户问题：
{question}
"""


def parse_tool_call_args(raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not raw_arguments:
        return {}
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}


def build_llm_planned_steps(
    question: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str] | None = None,
) -> list[AgentStep]:
    preferred_sources = preferred_sources or []
    client = agent.get_deepseek_client()
    if client is None:
        return []

    response = client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[
            {
                "role": "user",
                "content": build_planner_prompt(
                    question=question,
                    use_web=use_web,
                    top_k=top_k,
                    web_max_results=web_max_results,
                    preferred_sources=preferred_sources,
                ),
            }
        ],
        tools=PLANNER_TOOLS,
        tool_choice="auto",
        temperature=0,
        max_tokens=300,
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    steps: list[AgentStep] = []

    for tool_call in tool_calls:
        function = tool_call.function
        tool_name = function.name
        args = parse_tool_call_args(function.arguments)
        reason = args.pop("reason", "")

        if tool_name == "web_collect" and use_web:
            steps.append(
                AgentStep(
                    name="联网收集资料",
                    tool="web_collect",
                    reason=reason or "大模型判断需要联网补充资料。",
                    args={
                        "question": args.get("question", question),
                        "max_results": min(int(args.get("max_results", web_max_results)), web_max_results),
                    },
                )
            )
        elif tool_name == "rag_search":
            steps.append(
                AgentStep(
                    name="RAG 检索排序",
                    tool="rag_search",
                    reason=reason or "大模型判断需要从知识库检索证据资料。",
                    args={
                        "question": args.get("question", question),
                        "top_k": min(int(args.get("top_k", top_k)), top_k),
                        "preferred_sources": preferred_sources,
                    },
                )
            )

    return normalize_planned_steps(
        steps=steps,
        question=question,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
    )


def normalize_planned_steps(
    steps: list[AgentStep],
    question: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str],
) -> list[AgentStep]:
    tool_order = {"web_collect": 0, "rag_search": 1}
    allowed_steps = [
        step
        for step in steps
        if step.tool in tool_order and (step.tool != "web_collect" or use_web)
    ]
    deduped: dict[str, AgentStep] = {}
    for step in allowed_steps:
        deduped.setdefault(step.tool, step)

    if use_web and not preferred_sources and "web_collect" not in deduped:
        deduped["web_collect"] = AgentStep(
            name="联网收集资料",
            tool="web_collect",
            reason="没有用户上传资料，系统补充联网收集步骤。",
            args={"question": question, "max_results": web_max_results},
        )

    if "rag_search" not in deduped:
        deduped["rag_search"] = AgentStep(
            name="RAG 检索排序",
            tool="rag_search",
            reason="最终回答需要先检索证据资料，系统补充 RAG 检索步骤。",
            args={
                "question": question,
                "top_k": top_k,
                "preferred_sources": preferred_sources,
            },
        )

    planned_steps = sorted(deduped.values(), key=lambda step: tool_order[step.tool])
    planned_steps.append(
        AgentStep(
            name="生成最终回答",
            tool="generate_answer",
            reason="把筛选后的资料交给大模型生成带依据的回答。",
            args={"question": question, "search_results": "$rag_search"},
        )
    )
    return planned_steps


def plan_agent_steps(
    question: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str] | None = None,
) -> list[AgentStep]:
    if ENABLE_LLM_PLANNER:
        try:
            steps = build_llm_planned_steps(
                question=question,
                use_web=use_web,
                top_k=top_k,
                web_max_results=web_max_results,
                preferred_sources=preferred_sources,
            )
            if steps:
                return steps
        except Exception:
            pass

    return build_rule_based_steps(
        question=question,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
    )


def run_tool(step: AgentStep, state: dict[str, Any]) -> ToolResult:
    tool = TOOLS[step.tool]
    args = step.args.copy()

    if args.get("search_results") == "$rag_search":
        args["search_results"] = state.get("search_results", [])

    started_at = time.time()
    result = tool(**args)
    result.elapsed_ms = int((time.time() - started_at) * 1000)
    return result


def run_agent(
    question: str,
    use_web: bool = True,
    top_k: int = 3,
    web_max_results: int = 3,
    preferred_sources: list[str] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": question,
        "search_results": [],
        "answer": "",
    }
    trace: list[dict[str, Any]] = []
    steps = plan_agent_steps(
        question=question,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
    )

    for step in steps:
        try:
            result = run_tool(step, state)
            if step.tool == "rag_search":
                state["search_results"] = result.data
            elif step.tool == "generate_answer":
                state["answer"] = result.data
        except Exception as exc:
            result = ToolResult(
                status="failed",
                summary="执行失败。",
                error=str(exc),
            )
            trace.append(format_trace_item(step, result))
            raise

        trace.append(format_trace_item(step, result))

    return {
        "answer": state["answer"],
        "sources": state["search_results"],
        "steps": trace,
    }


def format_trace_item(step: AgentStep, result: ToolResult) -> dict[str, Any]:
    return {
        "name": step.name,
        "tool": step.tool,
        "reason": step.reason,
        "status": result.status,
        "summary": result.summary,
        "elapsed_ms": result.elapsed_ms,
        "error": result.error,
    }
