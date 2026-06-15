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


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str
    suggested_action: str = ""
    entities: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanResult:
    action: str
    reason: str
    confidence: float = 0.8
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    id: str
    name: str
    tool: str
    reason: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    output_key: str = ""
    timeout_ms: int = 10000
    retry: int = 0
    required: bool = True


@dataclass
class TaskGraph:
    nodes: list[TaskNode]


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


def classify_intent(question: str, preferred_sources: list[str]) -> IntentResult:
    stripped_question = question.strip()
    lowered_question = stripped_question.lower()
    entities = preferred_sources[:]
    constraints = {
        "has_uploads": bool(preferred_sources),
        "needs_freshness": False,
        "needs_upload_context": False,
        "needs_web_context": False,
    }

    if is_upload_status_question(stripped_question):
        return IntentResult(
            intent="upload_status",
            confidence=0.95,
            reason="用户在确认上传资料是否已经被系统看到。",
            suggested_action="read_upload_status",
            entities=entities,
            constraints=constraints,
        )

    chitchat_words = [
        "你好",
        "您好",
        "嗨",
        "hello",
        "hi",
        "我是",
        "认识一下",
        "你是谁",
        "介绍一下你自己",
        "你能做什么",
        "你会什么",
        "你擅长什么",
        "能帮我什么",
    ]
    if len(stripped_question) <= 30 and any(word in lowered_question for word in chitchat_words):
        return IntentResult(
            intent="chitchat",
            confidence=0.9,
            reason="用户输入更像寒暄、自我介绍或普通对话。",
            suggested_action="direct_answer",
            entities=entities,
            constraints=constraints,
        )

    latest_words = ["最近", "最新", "今天", "现在", "趋势", "新闻", "动态", "current", "latest"]
    if any(word in lowered_question for word in latest_words):
        constraints["needs_freshness"] = True
        constraints["needs_web_context"] = True
        return IntentResult(
            intent="latest_research",
            confidence=0.82,
            reason="用户问题涉及近期信息或外部动态，需要联网补充资料。",
            suggested_action="collect_context",
            entities=entities,
            constraints=constraints,
        )

    upload_qa_words = ["总结", "提取", "分析", "资料", "文档", "pdf", "文件", "这份"]
    if preferred_sources and any(word in lowered_question for word in upload_qa_words):
        constraints["needs_upload_context"] = True
        return IntentResult(
            intent="document_qa",
            confidence=0.82,
            reason="用户问题需要基于已上传资料回答。",
            suggested_action="collect_context",
            entities=entities,
            constraints=constraints,
        )

    return IntentResult(
        intent="general_qa",
        confidence=0.62,
        reason="未命中特定状态或闲聊意图，按通用问答处理。",
        suggested_action="collect_context",
        entities=entities,
        constraints=constraints,
    )


def plan_high_level_action(intent: IntentResult, preferred_sources: list[str], use_web: bool) -> PlanResult:
    if intent.intent == "chitchat":
        return PlanResult(
            action="direct_answer",
            reason="寒暄或普通对话不需要检索资料，直接回复即可。",
            confidence=0.9,
            params={"needs_web": False, "needs_upload": False},
        )

    if intent.intent == "upload_status":
        return PlanResult(
            action="read_upload_status",
            reason="用户要确认上传状态，直接读取应用状态。",
            confidence=0.95,
            params={"needs_web": False, "needs_upload": False},
        )

    if intent.intent == "latest_research" and use_web:
        return PlanResult(
            action="collect_context",
            reason="问题涉及最新外部信息，需要联网资料和本地资料共同进入上下文。",
            confidence=0.84,
            params={"needs_web": True, "needs_upload": bool(preferred_sources)},
        )

    if intent.intent == "document_qa" or preferred_sources:
        return PlanResult(
            action="collect_context",
            reason="问题需要基于上传资料或知识库资料回答，先收集上下文。",
            confidence=0.82,
            params={"needs_web": False, "needs_upload": bool(preferred_sources)},
        )

    return PlanResult(
        action="collect_context",
        reason="通用问题先收集可用上下文，再评估是否足够回答。",
        confidence=0.68,
        params={"needs_web": use_web, "needs_upload": bool(preferred_sources)},
    )


def orchestrate_action(
    action: str,
    question: str,
    intent: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str],
) -> list[AgentStep]:
    if action == "direct_answer":
        return [
            AgentStep(
                name="直接回复",
                tool="direct_answer",
                reason="高层动作判断无需检索资料。",
                args={"question": question},
            )
        ]

    if action == "read_upload_status":
        return [
            AgentStep(
                name="读取上传状态",
                tool="upload_status",
                reason="高层动作判断需要读取当前上传资料状态。",
                args={"preferred_sources": preferred_sources},
            )
        ]

    steps: list[AgentStep] = []
    should_collect_web = use_web and (intent == "latest_research" or not preferred_sources)
    if should_collect_web:
        steps.append(
            AgentStep(
                name="联网收集资料",
                tool="web_collect",
                reason="DAG 节点：收集外部网页资料并写入资料库。",
                args={"question": question, "max_results": web_max_results},
            )
        )

    steps.append(
        AgentStep(
            name="RAG 检索排序",
            tool="rag_search",
            reason="DAG 节点：对上传资料、网页资料和本地资料做统一检索排序。",
            args={
                "question": question,
                "top_k": top_k,
                "preferred_sources": preferred_sources,
            },
        )
    )
    return steps


def build_task_graph(
    plan: PlanResult,
    question: str,
    intent: IntentResult,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str],
) -> TaskGraph:
    if plan.action == "direct_answer":
        return TaskGraph(nodes=[
            TaskNode(
                id="direct_answer",
                name="直接回复",
                tool="direct_answer",
                reason="工作流模板：普通对话直接生成回复。",
                args={"question": question},
                output_key="answer",
            )
        ])

    if plan.action == "read_upload_status":
        return TaskGraph(nodes=[
            TaskNode(
                id="upload_status",
                name="读取上传状态",
                tool="upload_status",
                reason="工作流模板：读取当前已入库上传资料状态。",
                args={"preferred_sources": preferred_sources},
                output_key="answer",
            )
        ])

    nodes: list[TaskNode] = []
    should_collect_web = use_web and (
        intent.intent == "latest_research"
        or not preferred_sources
        or plan.params.get("needs_web", False)
    )

    if should_collect_web:
        nodes.append(
            TaskNode(
                id="web_collect",
                name="联网收集资料",
                tool="web_collect",
                reason="工作流模板：先收集外部网页资料，并写入资料库。",
                args={"question": question, "max_results": web_max_results},
                output_key="web_collect",
                retry=1,
                required=False,
            )
        )

    nodes.append(
        TaskNode(
            id="rag_search",
            name="RAG 检索排序",
            tool="rag_search",
            reason="工作流模板：对上传资料、网页资料和本地资料做统一检索排序。",
            args={
                "question": question,
                "top_k": top_k,
                "preferred_sources": preferred_sources,
            },
            depends_on=["web_collect"] if should_collect_web else [],
            output_key="rag_search",
            required=True,
        )
    )

    return TaskGraph(nodes=nodes)


def validate_task_graph(graph: TaskGraph) -> None:
    node_ids = {node.id for node in graph.nodes}
    if len(node_ids) != len(graph.nodes):
        raise ValueError("任务图存在重复节点 ID。")

    for node in graph.nodes:
        if node.tool not in TOOLS:
            raise ValueError(f"未知工具：{node.tool}")
        for dependency in node.depends_on:
            if dependency not in node_ids:
                raise ValueError(f"节点 {node.id} 依赖不存在的节点：{dependency}")

    visiting: set[str] = set()
    visited: set[str] = set()
    dependencies = {node.id: node.depends_on for node in graph.nodes}

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("任务图存在循环依赖。")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in dependencies[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_ids:
        visit(node_id)


def task_node_to_step(node: TaskNode) -> AgentStep:
    return AgentStep(
        name=node.name,
        tool=node.tool,
        reason=node.reason,
        args=node.args,
    )


def get_ready_nodes(graph: TaskGraph, completed: set[str], failed: set[str]) -> list[TaskNode]:
    return [
        node
        for node in graph.nodes
        if node.id not in completed
        and node.id not in failed
        and all(dependency in completed for dependency in node.depends_on)
    ]


def run_task_graph(graph: TaskGraph, state: dict[str, Any]) -> tuple[dict[str, ToolResult], list[dict[str, Any]]]:
    validate_task_graph(graph)
    completed: set[str] = set()
    failed: set[str] = set()
    results: dict[str, ToolResult] = {}
    trace: list[dict[str, Any]] = []
    batch_index = 1

    while len(completed) < len(graph.nodes):
        ready_nodes = get_ready_nodes(graph, completed, failed)
        if not ready_nodes:
            raise RuntimeError("没有可执行节点，可能存在前置节点失败或任务图依赖异常。")

        trace.append(
            make_stage_trace(
                name=f"DAG 执行批次 {batch_index}",
                tool="dag_runtime",
                reason="根据 depends_on 规则找出当前可执行节点；同一批次代表理论上可并发执行。",
                summary="本批节点：" + "、".join(node.id for node in ready_nodes),
            )
        )

        for node in ready_nodes:
            step = task_node_to_step(node)
            last_result: ToolResult | None = None
            for attempt in range(node.retry + 1):
                try:
                    last_result = run_tool(step, state)
                    break
                except Exception as exc:
                    last_result = ToolResult(
                        status="failed",
                        summary="节点执行失败。",
                        error=str(exc),
                    )
                    if attempt >= node.retry:
                        break

            result = last_result or ToolResult(status="failed", summary="节点没有返回结果。")
            results[node.output_key or node.id] = result
            results[node.id] = result

            if result.status == "success":
                completed.add(node.id)
                if node.output_key:
                    state[node.output_key] = result.data
                if node.tool == "rag_search":
                    state["search_results"] = result.data
                elif node.tool in {"direct_answer", "upload_status", "generate_answer"}:
                    state["answer"] = result.data
            else:
                failed.add(node.id)
                if node.required:
                    trace.append(format_trace_item(step, result))
                    raise RuntimeError(result.error or f"关键节点失败：{node.id}")
                completed.add(node.id)

            trace.append(format_trace_item(step, result))

        batch_index += 1

    return results, trace


def aggregate_context(tool_results: dict[str, ToolResult]) -> dict[str, Any]:
    search_results = tool_results.get("rag_search", ToolResult(status="success", summary="", data=[])).data or []
    web_results = tool_results.get("web_collect", ToolResult(status="success", summary="", data=[])).data or []
    normalized_results = normalize_context_items(search_results)
    deduped_results = dedupe_context_items(normalized_results)
    packed_results = apply_source_quota(deduped_results)
    source_counts = {
        "upload": sum(1 for item in packed_results if item.get("source_type") == "upload"),
        "web": sum(1 for item in packed_results if item.get("source_type") == "web"),
        "local": sum(1 for item in packed_results if item.get("source_type") == "local"),
        "image": sum(1 for item in packed_results if item.get("source_type") == "image"),
        "unknown": sum(1 for item in packed_results if item.get("source_type") == "unknown"),
    }
    return {
        "search_results": packed_results,
        "web_results": web_results,
        "source_counts": source_counts,
        "quality_signals": {
            "raw_count": len(search_results),
            "deduped_count": len(deduped_results),
            "packed_count": len(packed_results),
            "has_upload_context": source_counts["upload"] > 0,
            "has_web_context": source_counts["web"] > 0,
            "has_citations": all(bool(item.get("source")) for item in packed_results),
        },
    }


def evaluate_context(intent: str, aggregated: dict[str, Any]) -> dict[str, Any]:
    search_results = aggregated.get("search_results", [])
    source_counts = aggregated.get("source_counts", {})
    quality_signals = aggregated.get("quality_signals", {})
    has_sources = bool(search_results)
    has_upload_sources = source_counts.get("upload", 0) > 0
    has_web_sources = source_counts.get("web", 0) > 0
    has_citations = quality_signals.get("has_citations", False)
    missing_aspects: list[str] = []

    if intent == "document_qa":
        sufficient = has_upload_sources or has_sources
        if not has_upload_sources:
            missing_aspects.append("缺少上传资料命中")
        reason = "文档问答优先看上传资料；当前已命中上传资料。" if has_upload_sources else "未命中上传资料，使用现有检索资料兜底。"
    elif intent == "latest_research":
        sufficient = has_sources
        if not has_web_sources:
            missing_aspects.append("缺少网页或近期资料")
        reason = "最新信息问题需要外部或本地资料；当前已有可用检索结果。" if has_sources else "未检索到可用资料。"
    else:
        sufficient = has_sources
        reason = "通用问题已有检索资料。" if has_sources else "没有检索到可用资料。"

    if has_sources and not has_citations:
        missing_aspects.append("部分资料缺少可引用来源")

    next_action = "generate_answer" if sufficient else "broaden_search"
    confidence = 0.82 if sufficient else 0.45

    return {
        "sufficient": sufficient,
        "confidence": confidence,
        "reason": reason,
        "source_count": len(search_results),
        "missing_aspects": missing_aspects,
        "next_action": next_action,
    }


def normalize_context_items(search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, item in enumerate(search_results, start=1):
        copied = item.copy()
        copied.setdefault("source_type", "unknown")
        copied.setdefault("source", copied.get("url", f"资料{index}"))
        copied.setdefault("document", copied.get("text", ""))
        copied.setdefault("final_score", copied.get("score", 0))
        copied["context_item_id"] = (
            copied.get("id")
            or f"{copied.get('source_type')}:{copied.get('source')}:{copied.get('chunk_index', index)}"
        )
        normalized.append(copied)
    return normalized


def dedupe_context_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        content = item.get("document", "")
        key = "|".join([
            str(item.get("document_key", "")),
            str(item.get("source", "")),
            str(item.get("chunk_index", "")),
            content[:120],
        ])
        if key in seen_keys:
            item["aggregator_skip_reason"] = "重复资料"
            continue
        seen_keys.add(key)
        item["aggregator_skip_reason"] = ""
        deduped.append(item)
    return deduped


def apply_source_quota(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quotas = {
        "upload": 6,
        "web": 4,
        "local": 3,
        "image": 3,
        "unknown": 2,
    }
    counts: dict[str, int] = {}
    packed: list[dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda item: item.get("final_score", 0), reverse=True)

    for item in sorted_items:
        source_type = item.get("source_type", "unknown")
        quota = quotas.get(source_type, quotas["unknown"])
        if counts.get(source_type, 0) >= quota:
            item["aggregator_skip_reason"] = "来源配额已满"
            continue
        counts[source_type] = counts.get(source_type, 0) + 1
        item["aggregator_order"] = len(packed) + 1
        packed.append(item)

    return packed


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


def tool_direct_answer(question: str) -> ToolResult:
    client = agent.get_deepseek_client()
    if client is None:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY。")

    response = client.chat.completions.create(
        model=agent.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""请直接回复用户。
要求：
1. 如果是寒暄、自我介绍、普通对话，简洁自然地回应。
2. 不要声称自己检索了资料。
3. 不要编造参考来源。

用户输入：
{question}
""",
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )
    answer = response.choices[0].message.content
    agent.conversation_history.append({"role": "user", "content": question})
    agent.conversation_history.append({"role": "assistant", "content": answer})

    return ToolResult(
        status="success",
        summary="无需检索，已直接生成回复。",
        data=answer,
    )


def tool_upload_status(preferred_sources: list[str]) -> ToolResult:
    if preferred_sources:
        source_lines = "\n".join(f"- {source}" for source in preferred_sources)
        answer = f"能看到。你当前上传并入库的资料有：\n{source_lines}\n\n你可以直接问我总结、提取重点或围绕这些资料做分析。"
        summary = f"已读取上传状态，当前可见 {len(preferred_sources)} 个上传资料来源。"
    else:
        answer = "我目前没有看到已成功入库的上传资料。你可以先在左侧上传文件，等侧边栏出现“已入库资料”后再提问。"
        summary = "已读取上传状态，当前没有可见的上传资料来源。"

    return ToolResult(
        status="success",
        summary=summary,
        data=answer,
    )


TOOLS: dict[str, Callable[..., ToolResult]] = {
    "web_collect": tool_web_collect,
    "rag_search": tool_rag_search,
    "generate_answer": tool_generate_answer,
    "direct_answer": tool_direct_answer,
    "upload_status": tool_upload_status,
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
    {
        "type": "function",
        "function": {
            "name": "direct_answer",
            "description": "不检索资料，直接回复用户。适合寒暄、自我介绍、闲聊、简单确认、无需事实依据或无需使用上传资料的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户原始输入。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么这个问题不需要调用 RAG 或联网工具。",
                    },
                },
                "required": ["question", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_status",
            "description": "读取当前应用状态，告诉用户是否能看到已上传并入库的资料。适合用户问：能看到上传资料吗、有没有收到文件、这个资料你看得到吗。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "为什么需要读取上传状态，而不是检索资料内容。",
                    },
                },
                "required": ["reason"],
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
2. 如果是寒暄、自我介绍、闲聊、简单确认，调用 direct_answer，不要调用 rag_search。
3. 如果用户是在问“能不能看到上传资料、有没有收到文件、是否已上传成功”，调用 upload_status。
4. 如果用户问题需要依据上传资料、知识库资料或历史资料，调用 rag_search。
5. 如果问题涉及最新趋势、新闻、当前状态、外部事实，并且允许联网，先调用 web_collect，再调用 rag_search。
6. 如果问题明显只要求总结用户上传资料，直接调用 rag_search，不要调用 web_collect。
7. 如果已经调用 direct_answer 或 upload_status，不要再调用其他工具。
8. 不要调用 generate_answer，资料型问题的最终回答由系统在检索后统一生成。

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
        elif tool_name == "direct_answer":
            steps.append(
                AgentStep(
                    name="直接回复",
                    tool="direct_answer",
                    reason=reason or "大模型判断这个问题不需要检索资料。",
                    args={"question": args.get("question", question)},
                )
            )
        elif tool_name == "upload_status":
            steps.append(
                AgentStep(
                    name="读取上传状态",
                    tool="upload_status",
                    reason=reason or "大模型判断用户在确认上传资料是否可见。",
                    args={"preferred_sources": preferred_sources},
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
    if is_upload_status_question(question):
        return [
            AgentStep(
                name="读取上传状态",
                tool="upload_status",
                reason="用户在确认上传资料是否可见，直接读取当前应用状态。",
                args={"preferred_sources": preferred_sources},
            )
        ]

    tool_order = {"web_collect": 0, "rag_search": 1, "direct_answer": 2, "upload_status": 3}
    allowed_steps = [
        step
        for step in steps
        if step.tool in tool_order and (step.tool != "web_collect" or use_web)
    ]
    deduped: dict[str, AgentStep] = {}
    for step in allowed_steps:
        deduped.setdefault(step.tool, step)

    if "direct_answer" in deduped:
        return [deduped["direct_answer"]]

    if "upload_status" in deduped:
        return [deduped["upload_status"]]

    if use_web and not preferred_sources and "web_collect" not in deduped and "rag_search" in deduped:
        deduped["web_collect"] = AgentStep(
            name="联网收集资料",
            tool="web_collect",
            reason="资料型问题没有用户上传资料，系统补充联网收集步骤。",
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
    preferred_sources = preferred_sources or []
    if is_upload_status_question(question):
        return [
            AgentStep(
                name="读取上传状态",
                tool="upload_status",
                reason="用户在确认上传资料是否可见，直接读取当前应用状态。",
                args={"preferred_sources": preferred_sources},
            )
        ]

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


def is_upload_status_question(question: str) -> bool:
    upload_words = ["上传", "资料", "文件", "pdf", "文档"]
    status_words = [
        "看到",
        "看得到",
        "看不到",
        "看见",
        "看不见",
        "收到",
        "有没有",
        "能不能",
        "可以看到",
        "识别",
        "读取",
        "成功",
    ]
    lower_question = question.lower()
    return any(word in lower_question for word in upload_words) and any(
        word in lower_question for word in status_words
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
        "planner_mode": "llm_tool_calling" if ENABLE_LLM_PLANNER else "rule_based",
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
            elif step.tool in {"generate_answer", "direct_answer", "upload_status"}:
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
        "planner_mode": state["planner_mode"],
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


def make_stage_trace(
    name: str,
    tool: str,
    reason: str,
    summary: str,
    status: str = "success",
    elapsed_ms: int = 0,
    error: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "tool": tool,
        "reason": reason,
        "status": status,
        "summary": summary,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }


def validate_final_answer(answer: str, search_results: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    if not answer.strip():
        warnings.append("答案为空")
    if search_results and "参考" not in answer and "来源" not in answer:
        warnings.append("答案可能缺少参考来源说明")
    if not evaluation.get("sufficient", False) and "资料不足" not in answer:
        warnings.append("资料不足时未明确提示")

    return {
        "passed": not warnings,
        "warnings": warnings,
    }


def run_agent_pro(
    question: str,
    use_web: bool = True,
    top_k: int = 3,
    web_max_results: int = 3,
    preferred_sources: list[str] | None = None,
) -> dict[str, Any]:
    preferred_sources = preferred_sources or []
    trace: list[dict[str, Any]] = []
    tool_results: dict[str, ToolResult] = {}
    answer = ""
    search_results: list[dict[str, Any]] = []

    started_at = time.time()
    intent = classify_intent(question, preferred_sources)
    trace.append(
        make_stage_trace(
            name="意图分类",
            tool="intent_classifier",
            reason="先判断用户请求类型，避免所有问题都进入同一条 RAG 链路。",
            summary=f"识别为 {intent.intent}，置信度 {intent.confidence:.2f}。{intent.reason}",
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    )

    started_at = time.time()
    plan = plan_high_level_action(intent, preferred_sources, use_web)
    trace.append(
        make_stage_trace(
            name="高层规划",
            tool="planner",
            reason="根据意图选择业务级动作，而不是直接暴露所有底层工具。",
            summary=f"选择动作：{plan.action}。{plan.reason}",
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    )

    started_at = time.time()
    task_graph = build_task_graph(
        plan=plan,
        question=question,
        intent=intent,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
    )
    trace.append(
        make_stage_trace(
            name="任务编排",
            tool="orchestrator",
            reason="把高层动作展开成可执行 DAG 任务图，并写入节点依赖关系。",
            summary="执行节点：" + " → ".join(
                f"{node.id}(依赖:{','.join(node.depends_on) or '无'})"
                for node in task_graph.nodes
            ),
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    )

    state = {
        "question": question,
        "search_results": [],
        "answer": "",
        "planner_mode": "pro_runtime",
    }

    tool_results, runtime_trace = run_task_graph(task_graph, state)
    trace.extend(runtime_trace)
    search_results = state.get("search_results", [])
    answer = state.get("answer", "")

    if answer:
        return {
            "answer": answer,
            "sources": search_results,
            "steps": trace,
            "planner_mode": "pro_runtime",
        }

    started_at = time.time()
    aggregated = aggregate_context(tool_results)
    search_results = aggregated["search_results"]
    trace.append(
        make_stage_trace(
            name="结果聚合",
            tool="aggregator",
            reason="把多个工具或 DAG 节点的结果合并成统一上下文候选。",
            summary=(
                f"聚合到 {len(search_results)} 条检索资料；"
                f"原始 {aggregated['quality_signals']['raw_count']} 条，"
                f"去重后 {aggregated['quality_signals']['deduped_count']} 条，"
                f"上传 {aggregated['source_counts']['upload']} 条，"
                f"网页 {aggregated['source_counts']['web']} 条，"
                f"本地 {aggregated['source_counts']['local']} 条。"
            ),
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    )

    started_at = time.time()
    evaluation = evaluate_context(intent.intent, aggregated)
    trace.append(
        make_stage_trace(
            name="资料评估",
            tool="evaluator",
            reason="判断当前资料是否足够支撑最终回答。",
            summary=(
                f"资料是否足够：{'是' if evaluation['sufficient'] else '否'}；"
                f"资料数：{evaluation['source_count']}；"
                f"置信度：{evaluation['confidence']:.2f}；"
                f"建议动作：{evaluation['next_action']}。{evaluation['reason']}"
            ),
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    )

    final_step = AgentStep(
        name="生成最终回答",
        tool="generate_answer",
        reason="基于聚合并评估后的资料生成最终回答。",
        args={"question": question, "search_results": search_results},
    )
    final_result = run_tool(final_step, state)
    answer = final_result.data
    trace.append(format_trace_item(final_step, final_result))

    validation = validate_final_answer(answer, search_results, evaluation)
    trace.append(
        make_stage_trace(
            name="答案校验",
            tool="answer_validator",
            reason="对最终回答做规则化后验检查，包括空答案、引用提示和资料不足提示。",
            summary=(
                "校验通过。"
                if validation["passed"]
                else "发现提示：" + "；".join(validation["warnings"])
            ),
            status="success" if validation["passed"] else "warning",
        )
    )

    return {
        "answer": answer,
        "sources": search_results,
        "steps": trace,
        "planner_mode": "pro_runtime",
        "evaluation": evaluation,
        "validation": validation,
    }
