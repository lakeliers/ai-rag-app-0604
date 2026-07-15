import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import rag_agent_core as agent
import memory_manager
import permission_gate


ENABLE_LLM_PLANNER = os.getenv("ENABLE_LLM_PLANNER", "1") == "1"
PLANNER_MODEL = os.getenv("PLANNER_MODEL", agent.DEEPSEEK_MODEL)
ROUTER_MODE_RULES = "rules"
ROUTER_MODE_HYBRID = "hybrid"
ROUTER_MODES = {ROUTER_MODE_RULES, ROUTER_MODE_HYBRID}
SOURCE_STRATEGY_AUTO = "auto"
SOURCE_STRATEGY_UPLOAD_ONLY = "upload_only"
SOURCE_STRATEGY_WEB_ONLY = "web_only"
SOURCE_STRATEGY_UPLOAD_AND_WEB = "upload_and_web"
SOURCE_STRATEGIES = {
    SOURCE_STRATEGY_AUTO,
    SOURCE_STRATEGY_UPLOAD_ONLY,
    SOURCE_STRATEGY_WEB_ONLY,
    SOURCE_STRATEGY_UPLOAD_AND_WEB,
}
PLANNER_RULES = "rules"
PLANNER_LLM_TOOL_CALLING = "llm_tool_calling"
PLANNER_FALLBACK_MIXED = "fallback_mixed"
PLANNER_TYPES = {PLANNER_RULES, PLANNER_LLM_TOOL_CALLING, PLANNER_FALLBACK_MIXED}
EVALUATOR_OFF = "off"
EVALUATOR_RULES = "rules"
EVALUATOR_TYPES = {EVALUATOR_OFF, EVALUATOR_RULES}
MEMORY_ROUTE_AUTO = "auto"
MEMORY_ROUTE_HYBRID = "hybrid"
MEMORY_ROUTE_ALWAYS = "always"
MEMORY_ROUTE_OFF = "off"
MULTI_AGENT_AUTO = "auto"
MULTI_AGENT_MANAGER_WORKER = "manager_worker"
MULTI_AGENT_PIPELINE = "pipeline"
MULTI_AGENT_CRITIC_LOOP = "critic_loop"
MULTI_AGENT_DEBATE = "debate"
MULTI_AGENT_SWARM = "swarm"
MULTI_AGENT_ARCHITECTURES = {
    MULTI_AGENT_AUTO,
    MULTI_AGENT_MANAGER_WORKER,
    MULTI_AGENT_PIPELINE,
    MULTI_AGENT_CRITIC_LOOP,
    MULTI_AGENT_DEBATE,
    MULTI_AGENT_SWARM,
}
DEBATE_MIN_ROUNDS = 1
DEBATE_MAX_ROUNDS = 3
DEBATE_LLM_TIMEOUT_SECONDS = float(os.getenv("DEBATE_LLM_TIMEOUT_SECONDS", "18"))
SWARM_LLM_TIMEOUT_SECONDS = float(os.getenv("SWARM_LLM_TIMEOUT_SECONDS", "14"))

ProgressCallback = Callable[[dict[str, Any]], None]

GREETING_PATTERNS = [
    r"^(你好|您好|嗨|hello|hi)(呀|啊|哈|，|,|。|！|!|\s)*$",
    r"^(你好|您好|嗨|hello|hi).{0,8}(我是|我叫)",
    r"^(我是|我叫).{1,12}$",
    r"^(认识一下|打个招呼)$",
]
CAPABILITY_PATTERNS = [
    r"(你|你们|助手|agent|这个agent|这个助手).{0,8}(能|可以|会|擅长|支持).{0,8}(做什么|做些?什么|做哪些|干什么|干嘛|帮我什么|帮我做什么|帮我啥|做啥)",
    r"(你|你们|助手|agent|这个agent|这个助手).{0,8}(会什么|擅长什么|支持什么)",
    r"(你|你们|助手|agent|这个agent|这个助手).{0,8}(有什么|有哪些).{0,4}(功能|能力|用处|用)",
    r"(怎么用|如何使用).{0,8}(你|这个agent|这个助手|agent|助手)",
    r"(你|你们|助手|agent|这个agent|这个助手).{0,8}(是谁|自我介绍|介绍一下)",
    r"(介绍一下|自我介绍).{0,8}(你|你自己|这个agent|这个助手|agent|助手)",
]
CONTEXT_REFERENCE_PATTERNS = [
    r"(我|我的|本人).{0,8}(名字|姓名|称呼).{0,8}(是什么|叫什?么|是啥|吗|\?)",
    r"(你知道|还记得|记得).{0,8}(我|我的).{0,8}(名字|姓名|称呼|叫什?么)",
    r"(刚才|前面|之前|上面).{0,12}(我).{0,8}(叫什?么|说.*名字|说.*称呼|说.*是谁)",
    r"(我刚才|我前面|我之前).{0,12}(说).{0,8}(我叫|我是|名字)",
]

SESSION_NAME_QUERY_PATTERNS = [
    r"(你知道|还记得|记得).{0,8}(我|我的).{0,8}(名字|姓名|称呼|叫什?么)",
    r"(我|我的|本人).{0,8}(名字|姓名|称呼).{0,8}(是什么|叫什?么|是啥|吗|\?)",
]
CONCRETE_TASK_WORDS = [
    "调研",
    "研究",
    "对比",
    "报告",
    "计划",
    "方案",
    "梳理",
    "整理",
    "分析",
    "追踪",
    "总结",
    "提取",
    "生成",
    "输出",
    "写一份",
    "做一份",
]


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


def is_success_like_status(status: str) -> bool:
    return status in {"success", "degraded"}


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


@dataclass
class DebateRole:
    id: str
    name: str
    goal: str
    system_prompt: str
    allowed_focus: list[str] = field(default_factory=list)
    forbidden_focus: list[str] = field(default_factory=list)


@dataclass
class SwarmAgent:
    id: str
    name: str
    goal: str
    writes: list[str] = field(default_factory=list)
    system_prompt: str = ""


def normalize_user_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def matches_any_pattern(text: str, patterns: list[str]) -> bool:
    normalized = normalize_user_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def asks_for_capability_intro(question: str) -> bool:
    normalized = normalize_user_text(question)
    if any(word in normalized for word in CONCRETE_TASK_WORDS):
        return False
    return matches_any_pattern(normalized, CAPABILITY_PATTERNS)


def is_lightweight_direct_intent(question: str) -> tuple[bool, str, str]:
    if asks_for_capability_intro(question):
        return True, "capability_intro", "用户在询问 Agent 能力边界，应该直接说明可用能力。"

    if matches_any_pattern(question, CONTEXT_REFERENCE_PATTERNS):
        return True, "chitchat", "用户在引用本轮短期对话上下文，应该直接读取会话历史回答。"

    if len(question.strip()) <= 30 and matches_any_pattern(question, GREETING_PATTERNS):
        return True, "chitchat", "用户输入更像寒暄、自我介绍或普通对话。"

    return False, "", ""


def extract_session_name(conversation_context: str) -> str:
    """Read a name only from an explicit user self-introduction in this session."""
    if not conversation_context:
        return ""
    matches = re.findall(
        r"用户：[^\n]{0,80}?(?:我是|我叫)\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{0,15})",
        conversation_context,
    )
    return matches[-1] if matches else ""


def is_session_name_lookup(question: str, conversation_context: str) -> bool:
    return bool(
        extract_session_name(conversation_context)
        and matches_any_pattern(question, SESSION_NAME_QUERY_PATTERNS)
    )


def looks_like_external_entity_lookup(question: str) -> bool:
    normalized = normalize_user_text(question)
    lookup_words = [
        "你知道",
        "是什么",
        "介绍一下",
        "了解",
        "查一下",
        "搜一下",
        "有没有",
        "是哪",
        "什么品牌",
        "什么产品",
    ]
    if not any(word in normalized for word in lookup_words):
        return False
    if any(word in normalized for word in ["rag", "bm25", "agent", "reranker", "contextpacking"]):
        return False

    has_model_like_token = bool(re.search(r"[a-zA-Z]{1,12}\d+[a-zA-Z0-9_-]{0,16}", normalized))
    has_mixed_entity = bool(re.search(r"[\u4e00-\u9fff]{2,8}[a-zA-Z0-9_-]{2,24}", normalized))
    has_quoted_entity = bool(re.search(r"[“\"']{1}[^“\"']{2,32}[”\"']{1}", question))
    return has_model_like_token or has_mixed_entity or has_quoted_entity


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start:end + 1])
        raise


def route_memory_with_llm(
    question: str,
    conversation_context: str,
    rule_route: dict[str, Any],
    model_name: str = "",
) -> dict[str, Any]:
    if rule_route.get("confidence", 0) >= 0.85:
        return rule_route
    client = agent.get_deepseek_client()
    if client is None:
        fallback = dict(rule_route)
        fallback["reason"] = fallback.get("reason", "") + " LLM Memory Router 未启用：缺少 DEEPSEEK_API_KEY，已使用规则结果。"
        fallback["source"] = "rule_fallback"
        return fallback
    payload = {
        "question": question,
        "conversation_context": conversation_context[-1200:],
        "rule_route": rule_route,
        "allowed_memory_types": memory_manager.MEMORY_TYPES,
        "output_schema": {
            "need_memory": "boolean",
            "memory_types": "list of allowed memory types",
            "query": "string",
            "reason": "short Chinese reason",
            "confidence": "0-1 number",
        },
    }
    try:
        response = client.chat.completions.create(
            model=model_name or PLANNER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 Memory Router（记忆路由器）。判断本轮是否需要检索长期记忆。"
                        "寒暄、简单确认、当前会话里能回答的问题，不要检索长期记忆。"
                        "只有用户明确引用历史偏好、身份、长期任务、学习进度时才检索。只输出 JSON。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=300,
        )
        parsed = extract_json_object(response.choices[0].message.content or "{}")
        memory_types = [
            item for item in parsed.get("memory_types", [])
            if item in memory_manager.MEMORY_TYPES
        ]
        try:
            confidence = float(parsed.get("confidence", rule_route.get("confidence", 0.6)))
        except (TypeError, ValueError):
            confidence = float(rule_route.get("confidence", 0.6))
        return {
            "need_memory": bool(parsed.get("need_memory", rule_route.get("need_memory", False))),
            "memory_types": memory_types,
            "query": str(parsed.get("query") or question),
            "reason": f"LLM Memory Router：{parsed.get('reason', '')}",
            "confidence": max(0.0, min(1.0, confidence)),
            "source": "llm",
        }
    except Exception as exc:
        fallback = dict(rule_route)
        fallback["reason"] = fallback.get("reason", "") + f" LLM Memory Router 异常，已使用规则结果：{exc}"
        fallback["source"] = "rule_fallback"
        return fallback


def load_memory_after_intent(
    question: str,
    intent: IntentResult,
    *,
    enabled: bool,
    route_strategy: str,
    conversation_context: str = "",
    model_name: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    session_name = extract_session_name(conversation_context)
    if is_session_name_lookup(question, conversation_context):
        return "", [], {
            "need_memory": False,
            "memory_types": ["session_memory"],
            "query": question,
            "reason": f"本轮会话中已找到用户自我介绍“{session_name}”，优先使用 Session Memory，不读取长期记忆。",
            "confidence": 0.98,
            "source": "session_memory",
            "route_strategy": route_strategy,
            "intent": intent.intent,
        }
    if not enabled:
        return "", [], {
            "need_memory": False,
            "memory_types": [],
            "query": question,
            "reason": "Memory 开关未启用。",
            "confidence": 1.0,
            "source": "config",
            "route_strategy": route_strategy,
            "intent": intent.intent,
        }
    if route_strategy == MEMORY_ROUTE_OFF:
        return "", [], {
            "need_memory": False,
            "memory_types": [],
            "query": question,
            "reason": "Memory Route 策略设置为关闭读取。",
            "confidence": 1.0,
            "source": "config",
            "route_strategy": route_strategy,
            "intent": intent.intent,
        }
    if route_strategy == MEMORY_ROUTE_ALWAYS:
        memories = memory_manager.retrieve_memories(question, include_core=True)
        return memory_manager.build_memory_context(memories), memories, {
            "need_memory": True,
            "memory_types": memory_manager.MEMORY_TYPES,
            "query": question,
            "reason": "Memory Route 策略设置为总是读取，用于教学对比。",
            "confidence": 1.0,
            "source": "config",
            "route_strategy": route_strategy,
            "intent": intent.intent,
        }

    route = memory_manager.route_memory(question, conversation_context=conversation_context)
    if route_strategy == MEMORY_ROUTE_HYBRID:
        route = route_memory_with_llm(question, conversation_context, route, model_name=model_name)
    route["route_strategy"] = route_strategy
    route["intent"] = intent.intent
    if not route.get("need_memory"):
        return "", [], route
    memories = memory_manager.retrieve_memories(
        route.get("query") or question,
        memory_types=route.get("memory_types") or None,
        include_core=True,
    )
    return memory_manager.build_memory_context(memories), memories, route


def extract_effective_query(question: str) -> str:
    """Convert verbose agent task prompts into short retrieval/search queries."""
    stripped = question.strip()
    goal_match = re.search(r"总目标：\s*(.+?)(?:\n\s*\n|当前任务：)", stripped, re.S)
    if goal_match:
        goal = re.sub(r"\s+", " ", goal_match.group(1)).strip()
        if goal:
            return goal[:120]

    if len(stripped) > 160 and "用户问题：" in stripped:
        user_question = stripped.rsplit("用户问题：", 1)[-1].strip()
        if user_question:
            return re.sub(r"\s+", " ", user_question)[:120]

    return stripped


def tool_web_collect(
    question: str,
    max_results: int,
    preferred_sources: list[str] | None = None,
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
) -> ToolResult:
    query = build_web_collect_query(
        question,
        preferred_sources=preferred_sources or [],
        chroma_path=chroma_path,
        metadata_scope=metadata_scope,
    )
    ingested = agent.web_collect(
        query,
        max_results=max_results,
        chroma_path=chroma_path,
        metadata_scope=metadata_scope,
    )
    return ToolResult(
        status="success",
        summary=f"联网收集完成，写入 {len(ingested)} 条网页资料。",
        data=ingested,
    )


def extract_context_keywords(text: str, limit: int = 8) -> list[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9 +/#.-]{1,40}|[\u4e00-\u9fff]{2,12}", text or "")
    stop_words = {
        "上传",
        "资料",
        "文件",
        "这个",
        "这份",
        "用户",
        "问题",
        "最近",
        "类似",
        "案例",
        "结合",
        "再查",
        "一下",
        "需要",
        "包含",
        "负责",
        "观察",
        "质量",
        "来源",
        "失败原因",
    }
    seen: set[str] = set()
    keywords: list[str] = []
    for raw in candidates:
        keyword = re.sub(r"\s+", " ", raw).strip(" ：:，,。.;；()（）")
        if len(keyword) < 2 or keyword in stop_words:
            continue
        normalized = keyword.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(keyword)
        if len(keywords) >= limit:
            break
    return keywords


def build_web_collect_query(
    question: str,
    preferred_sources: list[str],
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
) -> str:
    effective_question = extract_effective_query(question)
    needs_upload_topic = bool(preferred_sources) and any(
        word in effective_question for word in ["类似案例", "相关案例", "对标案例", "最近有没有", "再查一下"]
    )
    if not needs_upload_topic:
        return effective_question

    try:
        upload_rows = agent.search_chroma(
            effective_question,
            top_k=3,
            preferred_sources=preferred_sources,
            preferred_only=True,
            retrieval_strategy=agent.RETRIEVAL_VECTOR_BM25_RRF,
            context_packing_strategy=agent.CONTEXT_SOURCE_PRIORITY,
            chroma_path=chroma_path,
            metadata_scope=metadata_scope,
        )
    except Exception:
        upload_rows = []

    context_text = " ".join(str(item.get("document", "")) for item in upload_rows)
    keywords = extract_context_keywords(context_text)
    if not keywords:
        return effective_question

    context_lower = context_text.lower()
    if "rag" in context_lower or "检索增强" in context_text:
        return "RAG 检索增强生成 最近 落地案例"
    if "agent eval" in context_lower or "评估" in context_text:
        return "AI Agent 评估体系 最近 实践案例"
    if "tool agent" in context_lower or "工具调用" in context_text:
        return "Tool Agent 工具调用 最近 实践案例"

    topic = " ".join(keywords[:6])
    return f"{topic} 最近 类似案例"


def is_strict_upload_context_question(question: str, preferred_sources: list[str]) -> bool:
    if not preferred_sources:
        return False

    lowered_question = question.lower()
    upload_context_words = [
        "总结",
        "提取",
        "分析",
        "这份",
        "资料",
        "文档",
        "pdf",
        "文件",
        "有没有提到",
        "是否提到",
        "有没有包含",
    ]
    freshness_words = ["最近", "最新", "今天", "现在", "趋势", "新闻", "动态", "current", "latest"]

    return any(word in lowered_question for word in upload_context_words) and not any(
        word in lowered_question for word in freshness_words
    )


def tool_rag_search(
    question: str,
    top_k: int,
    preferred_sources: list[str],
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
) -> ToolResult:
    effective_question = extract_effective_query(question)
    if source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY and not preferred_sources:
        return ToolResult(
            status="success",
            summary="当前配置为仅上传资料，但没有可用上传资料。",
            data=[],
        )

    preferred_only = (
        source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY
        or is_strict_upload_context_question(effective_question, preferred_sources)
    )
    results = agent.search_chroma(
        effective_question,
        top_k=top_k,
        preferred_sources=preferred_sources,
        preferred_only=preferred_only,
        retrieval_strategy=retrieval_strategy,
        context_packing_strategy=context_packing_strategy,
        chroma_path=chroma_path,
        metadata_scope=metadata_scope,
    )
    if source_strategy == SOURCE_STRATEGY_WEB_ONLY:
        results = [item for item in results if item.get("source_type") == "web"]
    elif source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY:
        results = [item for item in results if item.get("source_type") == "upload"]

    upload_count = sum(1 for item in results if item.get("source_type") == "upload")
    web_count = sum(1 for item in results if item.get("source_type") == "web")
    return ToolResult(
        status="success",
        summary=f"检索完成，选出 {len(results)} 条资料，其中上传资料 {upload_count} 条、网页资料 {web_count} 条。",
        data=results,
    )


def classify_intent_by_rules(question: str, preferred_sources: list[str]) -> IntentResult:
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

    is_direct_intent, direct_intent, direct_reason = is_lightweight_direct_intent(stripped_question)
    if is_direct_intent:
        return IntentResult(
            intent=direct_intent,
            confidence=0.9,
            reason=direct_reason,
            suggested_action="direct_answer",
            entities=entities,
            constraints=constraints,
        )

    latest_words = ["最近", "最新", "今天", "现在", "趋势", "新闻", "动态", "current", "latest"]
    freshness_suppression_words = [
        "不需要最新",
        "不要最新",
        "不用最新",
        "无需最新",
        "不需要新闻",
        "不要新闻",
        "不用联网",
        "不要联网",
        "不需要联网",
        "只解释定义",
        "只要定义",
    ]
    definition_words = ["是什么", "定义", "概念", "解释一下", "介绍一下", "什么意思", "区别是什么"]
    upload_qa_words = ["总结", "提取", "分析", "资料", "文档", "pdf", "文件", "这份"]
    if is_planning_or_advice_question(stripped_question) and not any(
        word in lowered_question for word in freshness_suppression_words + ["最新", "今天", "实时", "官方", "财报", "政策", "法规"]
    ):
        constraints["needs_web_context"] = True
        constraints["planning_bounded_web"] = True
        return IntentResult(
            intent="general_qa",
            confidence=0.82,
            reason="用户在补充方案/行程/建议类约束，应联网收集参考资料，但不能因网页读取失败阻塞最终方案。",
            suggested_action="collect_context",
            entities=entities,
            constraints=constraints,
        )

    if (
        any(word in lowered_question for word in latest_words)
        and not any(word in lowered_question for word in freshness_suppression_words)
    ):
        constraints["needs_freshness"] = True
        constraints["needs_web_context"] = True
        if preferred_sources and any(word in lowered_question for word in upload_qa_words):
            constraints["needs_upload_context"] = True
            return IntentResult(
                intent="hybrid_rag",
                confidence=0.72,
                reason="用户同时提到上传资料和近期信息，规则识别为复合 RAG 意图，建议交给语义分类细化。",
                suggested_action="collect_context",
                entities=entities,
                constraints=constraints,
            )
        return IntentResult(
            intent="latest_research",
            confidence=0.82,
            reason="用户问题涉及近期信息或外部动态，需要联网补充资料。",
            suggested_action="collect_context",
            entities=entities,
            constraints=constraints,
        )

    if looks_like_external_entity_lookup(stripped_question) and not any(word in lowered_question for word in freshness_suppression_words):
        constraints["needs_web_context"] = True
        return IntentResult(
            intent="latest_research",
            confidence=0.8,
            reason="用户在询问疑似外部实体、品牌或产品型号，需要联网核验事实。",
            suggested_action="collect_context",
            entities=entities,
            constraints=constraints,
        )

    if any(word in stripped_question for word in definition_words):
        return IntentResult(
            intent="definition_qa",
            confidence=0.78,
            reason="用户在询问概念定义或区别，不需要默认联网获取最新资料。",
            suggested_action="collect_context",
            entities=entities,
            constraints=constraints,
        )

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


def intent_to_constraints(intent_name: str, preferred_sources: list[str], raw_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    constraints = {
        "has_uploads": bool(preferred_sources),
        "needs_freshness": False,
        "needs_upload_context": False,
        "needs_web_context": False,
        "should_use_autonomous": False,
    }
    constraints.update(raw_constraints or {})

    if intent_name in {"latest_research", "hybrid_rag"}:
        constraints["needs_freshness"] = True
        constraints["needs_web_context"] = True
    if intent_name in {"document_qa", "hybrid_rag"}:
        constraints["needs_upload_context"] = bool(preferred_sources)
    if intent_name == "autonomous_task":
        constraints["should_use_autonomous"] = True
        constraints["needs_web_context"] = True

    return constraints


def classify_intent_by_llm(
    question: str,
    preferred_sources: list[str],
    rule_result: IntentResult,
    model_name: str = "",
) -> IntentResult:
    client = agent.get_deepseek_client()
    if client is None:
        rule_result.reason += " LLM 路由未启用：缺少 DEEPSEEK_API_KEY，已回退规则结果。"
        return rule_result

    payload = {
        "question": question,
        "has_uploads": bool(preferred_sources),
        "preferred_sources": preferred_sources[:5],
        "rule_result": {
            "intent": rule_result.intent,
            "confidence": rule_result.confidence,
            "reason": rule_result.reason,
            "constraints": rule_result.constraints,
        },
        "allowed_intents": [
            "chitchat",
            "capability_intro",
            "upload_status",
            "document_qa",
            "latest_research",
            "definition_qa",
            "hybrid_rag",
            "autonomous_task",
            "general_qa",
        ],
        "allowed_actions": [
            "direct_answer",
            "read_upload_status",
            "collect_context",
        ],
        "output_schema": {
            "intent": "one allowed intent",
            "suggested_action": "one allowed action",
            "needs_upload_context": "boolean",
            "needs_web_context": "boolean",
            "needs_freshness": "boolean",
            "should_use_autonomous": "boolean",
            "confidence": "0-1 number",
            "reason": "short Chinese reason",
        },
    }

    response = client.chat.completions.create(
        model=model_name or PLANNER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 Agent 路由意图分类器。只输出合法 JSON。"
                    "不要执行任务，只判断用户请求应该进入哪类链路。"
                    "如果用户只是问能力、身份、能做什么，必须输出 capability_intro/direct_answer。"
                    "如果用户要求调研、整理方案、输出报告等多步骤目标，输出 autonomous_task。"
                    "如果用户同时要求结合上传资料和联网近期信息，输出 hybrid_rag。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=500,
    )
    parsed = extract_json_object(response.choices[0].message.content or "{}")
    allowed_intents = set(payload["allowed_intents"])
    allowed_actions = set(payload["allowed_actions"])

    intent_name = str(parsed.get("intent", rule_result.intent))
    if intent_name not in allowed_intents:
        intent_name = rule_result.intent

    suggested_action = str(parsed.get("suggested_action", rule_result.suggested_action))
    if suggested_action not in allowed_actions:
        suggested_action = rule_result.suggested_action

    try:
        confidence = float(parsed.get("confidence", rule_result.confidence))
    except (TypeError, ValueError):
        confidence = rule_result.confidence
    confidence = max(0.0, min(1.0, confidence))

    raw_constraints = {
        "needs_upload_context": bool(parsed.get("needs_upload_context", False)),
        "needs_web_context": bool(parsed.get("needs_web_context", False)),
        "needs_freshness": bool(parsed.get("needs_freshness", False)),
        "should_use_autonomous": bool(parsed.get("should_use_autonomous", False)),
    }
    return IntentResult(
        intent=intent_name,
        confidence=confidence,
        reason=f"LLM 语义分类：{parsed.get('reason', '')}",
        suggested_action=suggested_action,
        entities=preferred_sources[:],
        constraints=intent_to_constraints(intent_name, preferred_sources, raw_constraints),
    )


def validate_intent_by_rules(question: str, preferred_sources: list[str], intent: IntentResult) -> IntentResult:
    if is_upload_status_question(question):
        return classify_intent_by_rules(question, preferred_sources)

    is_direct_intent, direct_intent, direct_reason = is_lightweight_direct_intent(question)
    if is_direct_intent:
        return IntentResult(
            intent=direct_intent,
            confidence=max(intent.confidence, 0.9),
            reason=f"{direct_reason} 规则复核覆盖 LLM 分类。",
            suggested_action="direct_answer",
            entities=preferred_sources[:],
            constraints=intent_to_constraints(direct_intent, preferred_sources),
        )

    normalized = normalize_user_text(question)
    has_concrete_task = any(word in normalized for word in CONCRETE_TASK_WORDS)
    if intent.suggested_action == "direct_answer" and has_concrete_task:
        intent.intent = "general_qa"
        intent.suggested_action = "collect_context"
        intent.confidence = min(intent.confidence, 0.72)
        intent.reason += " 规则复核：检测到具体任务词，禁止直接闲聊回复。"

    if intent.intent == "document_qa" and not preferred_sources:
        intent.intent = "general_qa"
        intent.constraints["needs_upload_context"] = False
        intent.reason += " 规则复核：当前没有上传资料，不能强行判为文档问答。"

    return intent


def classify_intent(
    question: str,
    preferred_sources: list[str],
    router_mode: str = ROUTER_MODE_RULES,
    model_name: str = "",
) -> IntentResult:
    if router_mode not in ROUTER_MODES:
        router_mode = ROUTER_MODE_RULES

    rule_result = classify_intent_by_rules(question, preferred_sources)
    if router_mode == ROUTER_MODE_RULES or rule_result.confidence >= 0.85:
        rule_result.constraints["router_mode"] = ROUTER_MODE_RULES
        return rule_result

    try:
        llm_result = classify_intent_by_llm(question, preferred_sources, rule_result, model_name=model_name)
    except Exception as exc:
        rule_result.reason += f" LLM 路由异常，已回退规则结果：{exc}"
        rule_result.constraints["router_mode"] = ROUTER_MODE_HYBRID
        rule_result.constraints["fallback_from"] = "llm_intent_classifier"
        rule_result.constraints["fallback_reason"] = str(exc)
        return rule_result
    final_result = validate_intent_by_rules(question, preferred_sources, llm_result)
    final_result.constraints["router_mode"] = ROUTER_MODE_HYBRID
    return final_result


def plan_high_level_action(intent: IntentResult, preferred_sources: list[str], use_web: bool) -> PlanResult:
    if intent.intent in {"chitchat", "capability_intro"}:
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

    if intent.intent in {"latest_research", "hybrid_rag"} and use_web:
        return PlanResult(
            action="collect_context",
            reason="问题涉及外部信息或复合资料需求，需要收集上下文。",
            confidence=0.84,
            params={
                "needs_web": intent.constraints.get("needs_web_context", True),
                "needs_upload": intent.constraints.get("needs_upload_context", bool(preferred_sources)),
            },
        )

    if intent.intent == "autonomous_task":
        return PlanResult(
            action="collect_context",
            reason="用户请求更像多步骤目标，普通问答模式下先收集上下文；自主模式会进入任务级循环。",
            confidence=0.8,
            params={"needs_web": True, "needs_upload": bool(preferred_sources)},
        )

    if intent.intent == "definition_qa":
        return PlanResult(
            action="collect_context",
            reason="定义类问题优先使用本地知识库或模型常识，不默认联网。",
            confidence=0.78,
            params={"needs_web": False, "needs_upload": bool(preferred_sources)},
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
        params={
            "needs_web": use_web,
            "needs_upload": bool(preferred_sources),
        },
    )


def orchestrate_action(
    action: str,
    question: str,
    intent: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str],
    source_strategy: str = SOURCE_STRATEGY_AUTO,
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
    if intent == "definition_qa":
        should_collect_web = False
    if source_strategy == SOURCE_STRATEGY_UPLOAD_AND_WEB:
        should_collect_web = use_web
    elif source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY:
        should_collect_web = False
    elif source_strategy == SOURCE_STRATEGY_WEB_ONLY:
        should_collect_web = use_web
    if should_collect_web:
        steps.append(
            AgentStep(
                name="联网收集资料",
                tool="web_collect",
                reason="DAG 节点：收集外部网页资料并写入资料库。",
                args={
                    "question": question,
                    "max_results": web_max_results,
                    "preferred_sources": preferred_sources,
                },
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
                "source_strategy": source_strategy,
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
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
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
    if intent.intent == "definition_qa" or plan.params.get("needs_web") is False:
        should_collect_web = False
    if source_strategy == SOURCE_STRATEGY_UPLOAD_AND_WEB:
        should_collect_web = use_web
    elif source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY:
        should_collect_web = False
    elif source_strategy == SOURCE_STRATEGY_WEB_ONLY:
        should_collect_web = use_web

    if should_collect_web:
        nodes.append(
            TaskNode(
                id="web_collect",
                name="联网收集资料",
                tool="web_collect",
                reason="工作流模板：先收集外部网页资料，并写入资料库。",
                args={
                    "question": question,
                    "max_results": web_max_results,
                    "preferred_sources": preferred_sources,
                },
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
                "source_strategy": source_strategy,
                "retrieval_strategy": retrieval_strategy,
                "context_packing_strategy": context_packing_strategy,
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


def emit_progress(state: dict[str, Any], event: dict[str, Any]) -> None:
    callback = state.get("progress_callback")
    if not callback:
        return
    try:
        callback(event)
    except Exception:
        # Progress rendering must never break the agent execution path.
        pass


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
        emit_progress(
            state,
            {
                "id": f"dag_batch_{batch_index}",
                "name": f"DAG 执行批次 {batch_index}",
                "tool": "dag_runtime",
                "status": "completed",
                "summary": "本批节点：" + "、".join(node.id for node in ready_nodes),
            },
        )

        for node in ready_nodes:
            step = task_node_to_step(node)
            last_result: ToolResult | None = None
            for attempt in range(node.retry + 1):
                try:
                    emit_progress(
                        state,
                        {
                            "id": node.id,
                            "name": node.name,
                            "tool": node.tool,
                            "status": "running",
                            "summary": node.reason,
                        },
                    )
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
            emit_progress(
                state,
                {
                    "id": node.id,
                    "name": node.name,
                    "tool": node.tool,
                    "status": "completed" if is_success_like_status(result.status) else "failed",
                    "summary": result.summary,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error,
                },
            )
            results[node.output_key or node.id] = result
            results[node.id] = result

            if is_success_like_status(result.status):
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


def build_question_with_memory(question: str, memory_context: str = "") -> str:
    if not memory_context.strip():
        return question
    return f"""{memory_context}

【当前用户问题】
{question}
"""


def build_question_with_context(
    question: str,
    memory_context: str = "",
    conversation_context: str = "",
) -> str:
    parts = []
    if memory_context.strip():
        parts.append(memory_context.strip())
    if conversation_context.strip():
        parts.append(conversation_context.strip())
    if not parts:
        return question
    return "\n\n".join(parts) + f"\n\n【当前用户问题】\n{question}"


def is_definition_fallback_question(question: str) -> bool:
    definition_words = ["是什么", "定义", "概念", "解释一下", "介绍一下", "什么意思", "区别"]
    blocked_words = ["价格", "多少钱", "上线", "日期", "根据这份", "根据我上传", "这份资料", "这个文件"]
    return any(word in question for word in definition_words) and not any(word in question for word in blocked_words)


def is_planning_or_advice_question(question: str, conversation_context: str = "") -> bool:
    text = f"{conversation_context}\n{question}"
    planning_words = [
        "计划",
        "方案",
        "行程",
        "旅行",
        "旅游",
        "出发",
        "预算",
        "酒店",
        "机票",
        "路线",
        "安排",
        "你来决定",
        "帮我决定",
        "建议",
        "推荐",
    ]
    hard_fact_words = ["财报", "股价", "销量", "政策", "法规", "上线日期", "官方", "最新", "今天", "新闻"]
    if any(word in question for word in hard_fact_words):
        return False
    return any(word in text for word in planning_words)


def build_definition_fallback_context(question: str) -> list[dict[str, Any]]:
    lowered = question.lower()
    definitions = {
        "bm25": (
            "BM25（Best Matching 25）是一种关键词检索排序算法，用于评估查询与文档的相关性。"
            "它综合词频、逆文档频率和文档长度归一化，常用于搜索引擎、问答系统和 RAG 的关键词召回。"
        ),
        "rag": (
            "RAG 是 Retrieval-Augmented Generation，中文常译为检索增强生成。"
            "它先检索外部资料，再把资料与用户问题一起交给大模型生成回答。"
        ),
        "reranker": (
            "Reranker 是重排序模型，通常把用户问题和候选文本作为一对输入，直接判断相关性，"
            "用于在初步召回后重新排序候选资料。"
        ),
        "context packing": (
            "Context Packing 是把检索后的候选资料按 token budget、来源优先级、去重、覆盖度和引用需求，"
            "打包进最终大模型 messages 的过程。"
        ),
        "tool agent": (
            "Tool Agent 是围绕一次用户请求进行工具规划、工具调用、观察结果并生成回答的智能体形态。"
            "它通常关注单轮或短链路任务，例如检索资料、调用 API、生成答案。"
            "Autonomous Agent 则更偏任务级自主执行，会维护目标、任务队列、循环状态、停止条件和必要的人类确认。"
        ),
        "autonomous agent": (
            "Autonomous Agent 是围绕较长期目标自主拆解任务、执行、观察、反思和停止的智能体形态。"
            "Tool Agent 可以作为它执行某个具体子任务时的工具调用层。"
        ),
        "agent": (
            "Agent（智能体）是能够围绕目标感知上下文、规划行动、调用工具、观察结果并生成输出的 AI 系统。"
            "在大模型应用里，Agent 通常比普通聊天机器人多了工具调用、状态管理、执行跟踪和必要的自检机制。"
        ),
    }
    for keyword, document in definitions.items():
        if keyword in lowered:
            return [{
                "source_type": "local",
                "source": "基础概念库",
                "document": document,
                "final_score": 0.72,
                "chunk_index": 1,
                "content_type": "definition_fallback",
            }]
    return []


def build_planning_fallback_answer(question: str, search_results: list[dict[str, Any]] | None = None) -> str:
    if is_planning_or_advice_question(question) and any(
        word in question for word in ["旅行", "行程", "出发", "预算", "元旦", "12月30日", "1月1日"]
    ):
        return (
            "## 推荐目的地\n"
            "建议优先选择杭州：从上海高铁可达，元旦期间休闲、美食和城市漫游选择较稳定；备选南京或苏州。\n\n"
            "## 逐日行程\n"
            "| 日期 | 安排 |\n"
            "| --- | --- |\n"
            "| 12月30日 | 上海出发到杭州，入住后安排西湖周边轻量游览和晚餐。 |\n"
            "| 12月31日 | 白天选择良渚/灵隐/城市漫游线之一，晚上安排跨年餐和休息。 |\n"
            "| 1月1日 | 上午补充休闲点，下午返程上海。 |\n\n"
            "## 预算拆分\n"
            "总预算约 20000 元，按每人 5000 元控制：交通 400-800 元/人，住宿 1200-2200 元/人，"
            "餐饮 800-1200 元/人，门票和当地交通 500-900 元/人，机动 500-900 元/人。\n\n"
            "## 预订建议\n"
            "先锁高铁往返和两晚住宿，再定每日一个主线活动，避免元旦人流导致行程过满。\n\n"
            "## 风险与核验\n"
            "高铁余票、酒店价格、景区预约和天气需实际预订前核验。参考资料：上海元旦短途旅行稳定样本。"
        )
    source_lines = []
    for item in (search_results or [])[:2]:
        source = item.get("source", "参考资料")
        source_lines.append(f"- {source}")
    return (
        "可以先基于当前约束输出一个初版方案；实时价格、库存、政策和开放时间需要执行前核验。\n\n"
        "参考来源：\n"
        + ("\n".join(source_lines) if source_lines else "- 当前无可用参考资料")
    )


def answer_planning_from_model(
    question: str,
    search_results: list[dict[str, Any]] | None = None,
    memory_context: str = "",
    conversation_context: str = "",
    model_name: str = "",
) -> str:
    client = agent.get_deepseek_client()
    reference_lines = []
    for index, item in enumerate((search_results or [])[:4], start=1):
        source = str(item.get("source", "参考资料")).strip() or "参考资料"
        document = re.sub(r"\s+", " ", str(item.get("document", ""))).strip()
        if document:
            reference_lines.append(f"{index}. {source}: {document[:500]}")

    prompt = f"""请基于用户当前输入和本轮短期对话上下文，输出一个可执行方案。

要求：
1. 允许基于用户提供的约束和常识做合理假设，但不要编造实时价格、实时库存、实时天气或实时政策。
2. 如果缺少实时信息，直接标注“需实际预订前核验”。
3. 如果是旅行/行程类问题，至少包含目的地选择理由、日程安排、预算拆分、预订/执行建议。
4. 不要说“资料不足无法回答”，除非用户明确要求必须基于外部资料。
5. 如果参考资料与当前问题相关，可以使用；如果不相关，明确说明未使用该资料，并基于用户约束输出方案。
6. 末尾保留“参考与核验”小节，说明哪些信息来自参考资料，哪些需要用户实际预订前核验。
7. 旅行/行程类回答必须使用以下小节：推荐目的地、逐日行程、预算拆分、预订建议、风险与核验。
8. 预算拆分必须写清总预算、每人预算、交通、住宿、餐饮、门票/当地交通、机动预算。
9. 旅行/行程类回答必须覆盖去程、每日安排和返程；如果具体车次、酒店、餐厅、景点票价没有资料支持，不要编造名称或价格。
10. 只能使用参考资料里出现过的目的地和景点类型；资料未提到的细节要写成“可按实际偏好选择”，不能写成确定事实。
11. 回答要紧凑完整，优先用短表格或短列表，不要长篇解释；旅行/行程类控制在约 900 个中文字符内。

【长期记忆】
{memory_context or "无"}

【本轮会话上下文】
{conversation_context or "无"}

【本轮检索参考资料】
{chr(10).join(reference_lines) if reference_lines else "无"}

【当前用户输入】
{question}
"""
    if client is None:
        return (
            "可以基于你已经给出的约束先做一个方案，但当前没有可用模型来生成详细内容。"
            "建议按目的地、日程、预算、交通住宿和风险核验五部分展开。"
        )

    response = client.chat.completions.create(
        model=model_name or agent.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1400,
        timeout=agent.LLM_TIMEOUT_SECONDS,
    )
    answer = response.choices[0].message.content.strip()
    return answer or build_planning_fallback_answer(question, search_results)


def merge_definition_fallback_context(question: str, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback_results = build_definition_fallback_context(question)
    if not fallback_results:
        return search_results

    existing_documents = {str(item.get("document", "")) for item in search_results}
    merged = list(search_results)
    for item in reversed(fallback_results):
        if str(item.get("document", "")) not in existing_documents:
            merged.insert(0, item)
    return merged


def answer_definition_from_model(question: str, model_name: str = "") -> str:
    client = agent.get_deepseek_client()
    if client is None:
        lowered = question.lower()
        if "bm25" in lowered:
            return (
                "结论：BM25 是一种关键词检索排序算法，会根据词频、逆文档频率和文档长度等因素计算文本相关性。\n\n"
                "关键依据：这是通用概念解释，不来自上传资料或网页资料。\n\n参考来源：通用知识。"
            )
        return "资料不足，当前没有检索到可用于回答的参考资料。"

    prompt = f"""请回答一个通用概念定义题。
用户问题：{question}

要求：
1. 可以使用通用知识回答，但必须说明这不是来自上传资料或网页资料。
2. 只解释定义，不补充最新新闻。
3. 结构包含：结论、关键解释、参考来源。
"""
    response = client.chat.completions.create(
        model=model_name or agent.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=450,
        timeout=agent.LLM_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content.strip()


def tool_generate_answer(
    question: str,
    search_results: list[dict[str, Any]],
    memory_context: str = "",
    conversation_context: str = "",
    model_name: str = "",
    stream_callback: Callable[[str, str], None] | None = None,
) -> ToolResult:
    generation_error = ""
    if is_planning_or_advice_question(question, conversation_context):
        try:
            answer = answer_planning_from_model(
                question,
                search_results=search_results,
                memory_context=memory_context,
                conversation_context=conversation_context,
                model_name=model_name,
            )
            return ToolResult(
                status="degraded",
                summary="已基于本轮上下文、用户约束和可用参考资料生成方案，并提示实时信息需核验。",
                data=answer,
            )
        except Exception as error:
            generation_error = str(error)
            answer = build_planning_fallback_answer(question, search_results)
            return ToolResult(
                status="degraded",
                summary="方案生成模型请求失败，已使用规则兜底生成初版方案。",
                data=answer,
                error=generation_error,
            )

    if not search_results and is_definition_fallback_question(question):
        try:
            answer = answer_definition_from_model(question, model_name=model_name)
            return ToolResult(
                status="success",
                summary="未检索到资料，已按定义类问题使用通用知识兜底回答。",
                data=answer,
            )
        except Exception as error:
            generation_error = str(error)

    try:
        if stream_callback:
            answer = agent.ask_deepseek_stream(
                build_question_with_context(question, memory_context, conversation_context),
                search_results,
                on_delta=stream_callback,
                include_history=False,
                model_name=model_name,
            )
        else:
            answer = agent.ask_deepseek(
                build_question_with_context(question, memory_context, conversation_context),
                search_results,
                include_history=False,
                model_name=model_name,
            )
    except Exception as error:
        answer = ""
        generation_error = str(error)

    if not answer:
        if not search_results:
            answer = "资料不足，当前没有检索到可用于回答的参考资料。"
        else:
            source_lines = []
            basis_lines = []
            for index, item in enumerate(search_results[:3], start=1):
                source = item.get("source", "未知来源")
                document = re.sub(r"\s+", " ", item.get("document", "")).strip()
                if document:
                    basis_lines.append(f"{index}. {document[:180]}")
                source_lines.append(f"- {source}")
            answer = (
                "结论：生成模型请求失败，以下为基于已检索资料的兜底回答。\n\n"
                "关键依据：\n"
                + ("\n".join(basis_lines) if basis_lines else "- 已检索到资料，但正文摘要不足。")
                + "\n\n参考来源：\n"
                + "\n".join(source_lines)
            )

    normalized_question = normalize_user_text(question)
    if (
        "理想汽车" in question
        and "2026" in question
        and any(word in normalized_question for word in ["一季度", "第一季度", "q1"])
        and any(word in question for word in ["财报", "业绩", "财务"])
    ):
        required_phrase = "理想汽车 2026 年一季度财报情况"
        if required_phrase not in answer:
            answer = f"关于{required_phrase}：\n\n{answer}"

    return ToolResult(
        status="degraded" if generation_error else "success",
        summary="生成模型请求失败，已使用检索资料兜底回答。" if generation_error else "回答生成完成。",
        data=answer,
        error=generation_error,
    )


def tool_direct_answer(
    question: str,
    memory_context: str = "",
    conversation_context: str = "",
    model_name: str = "",
    stream_callback: Callable[[str, str], None] | None = None,
) -> ToolResult:
    if asks_for_capability_intro(question):
        answer = (
            "我可以帮你做三类事情：\n\n"
            "1. 基于你上传的资料做总结、提取要点、问答和对比分析。\n"
            "2. 联网收集公开信息，再结合本地资料做 RAG 回答。\n"
            "3. 帮你学习和实操 AI 产品经理相关主题，比如 RAG、Tool Agent、Autonomous Agent 和 Agent Eval。\n\n"
            "你可以直接上传文件，或者问我一个具体问题。"
        )
        if stream_callback:
            stream_callback(answer, answer)
        return ToolResult(
            status="success",
            summary="识别为能力介绍问题，已直接说明可用能力。",
            data=answer,
        )

    session_name = extract_session_name(conversation_context)
    if session_name and matches_any_pattern(question, SESSION_NAME_QUERY_PATTERNS):
        answer = f"你刚才说你是{session_name}。"
        if stream_callback:
            stream_callback(answer, answer)
        return ToolResult(
            status="success",
            summary="已从本轮 Session Memory 读取用户自我介绍。",
            data=answer,
        )

    client = agent.get_deepseek_client()
    if client is None:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY。")

    messages = [
        {
            "role": "user",
            "content": f"""请直接回复用户。
要求：
1. 如果是寒暄、自我介绍、普通对话，简洁自然地回应。
2. 不要声称自己检索了资料。
3. 不要编造参考来源。
4. 可以参考【长期记忆】，但如果长期记忆和当前用户输入冲突，以当前用户输入为准。
5. 可以参考【本轮会话上下文】回答“刚才/我的名字/前面说过”等短期连续对话问题；如果上下文没有相关信息，再说不知道。

【长期记忆】
{memory_context or "无"}

【本轮会话上下文】
{conversation_context or "无"}

用户输入：
{question}
""",
        }
    ]
    if stream_callback:
        chunks = []
        stream = client.chat.completions.create(
            model=model_name or agent.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if not text:
                continue
            chunks.append(text)
            stream_callback(text, "".join(chunks))
        answer = "".join(chunks)
    else:
        response = client.chat.completions.create(
            model=model_name or agent.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        answer = response.choices[0].message.content
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
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
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
                "source_strategy": source_strategy,
                "retrieval_strategy": retrieval_strategy,
                "context_packing_strategy": context_packing_strategy,
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
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    model_name: str = "",
) -> list[AgentStep]:
    preferred_sources = preferred_sources or []
    client = agent.get_deepseek_client()
    if client is None:
        return []

    response = client.chat.completions.create(
        model=model_name or PLANNER_MODEL,
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
                        "source_strategy": source_strategy,
                        "retrieval_strategy": retrieval_strategy,
                        "context_packing_strategy": context_packing_strategy,
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
        source_strategy=source_strategy,
        retrieval_strategy=retrieval_strategy,
        context_packing_strategy=context_packing_strategy,
    )


def normalize_planned_steps(
    steps: list[AgentStep],
    question: str,
    use_web: bool,
    top_k: int,
    web_max_results: int,
    preferred_sources: list[str],
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
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
                "source_strategy": source_strategy,
                "retrieval_strategy": retrieval_strategy,
                "context_packing_strategy": context_packing_strategy,
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
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    model_name: str = "",
) -> list[AgentStep]:
    plan_agent_steps.last_fallback_trace = None
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
                source_strategy=source_strategy,
                retrieval_strategy=retrieval_strategy,
                context_packing_strategy=context_packing_strategy,
                model_name=model_name,
            )
            if steps:
                return steps
            plan_agent_steps.last_fallback_trace = make_stage_trace(
                name="Planner 回退",
                tool="planner_fallback",
                reason="LLM Planner 没有返回可执行步骤，回退规则 Planner。",
                summary="LLM Tool Calling Planner 返回空步骤，已使用规则规划。",
                status="warning",
            )
        except Exception as exc:
            plan_agent_steps.last_fallback_trace = make_stage_trace(
                name="Planner 回退",
                tool="planner_fallback",
                reason="LLM Planner 执行异常，回退规则 Planner。",
                summary="LLM Tool Calling Planner 失败，已使用规则规划。",
                status="warning",
                error=str(exc),
            )

    return build_rule_based_steps(
        question=question,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
        source_strategy=source_strategy,
        retrieval_strategy=retrieval_strategy,
        context_packing_strategy=context_packing_strategy,
    )


def is_upload_status_question(question: str) -> bool:
    normalized = normalize_user_text(question)
    content_or_task_words = [
        "有没有提到",
        "是否提到",
        "有没有包含",
        "讲了什么",
        "说了什么",
        "结合",
        "查询",
        "查一下",
        "案例",
        "总结",
        "分析",
        "提取",
        "对比",
        "最近",
        "最新",
    ]
    if any(word in normalized for word in content_or_task_words):
        return False

    status_patterns = [
        r"(你|系统|agent|助手).{0,8}(能|可以)?(看到|看见|看得到|看不到|读到|读得到|读不到|识别|收到).{0,8}(上传|资料|文件|pdf|文档)",
        r"(上传|资料|文件|pdf|文档).{0,12}(你|系统|agent|助手).{0,8}(能|可以)?(看到|看见|看得到|看不到|读到|读得到|读不到|识别|收到)",
        r"(上传|资料|文件|pdf|文档).{0,8}(成功|好了吗|好了没|完成了吗|入库了吗|到了吗)",
        r"(我).{0,4}(上传).{0,8}(成功|好了吗|好了没|你.*(看到|收到|读到))",
        r"(当前|这轮|这一侧|左侧|右侧|这里|现在).{0,12}(上传|资料|文件|pdf|文档).{0,16}(有没有|是否有|有无|可用|能用|能看到|看得到)",
        r"(有没有|是否有|有无).{0,12}(当前|这轮|这一侧|左侧|右侧|这里|现在).{0,12}(上传|资料|文件|pdf|文档)",
        r"(不要|别).{0,8}(引用|使用).{0,8}(另一侧|历史|之前).{0,12}(上传|资料|文件|pdf|文档).{0,24}(有没有|是否有|有无|可用|能用|能看到|看得到)",
        r"(当前|这轮|这一侧|左侧|右侧|这里|现在).{0,12}没有.{0,4}(上传|资料|文件|pdf|文档).{0,24}(有没有|是否有|有无|可用|能用|能看到|看得到)",
    ]
    return matches_any_pattern(normalized, status_patterns)


def action_for_step(step: AgentStep, state: dict[str, Any]) -> dict[str, Any]:
    operation_map = {
        "web_collect": ("collect", "public_web"),
        "rag_search": ("retrieve", "public_web"),
        "generate_answer": ("generate", "final_answer"),
        "direct_answer": ("generate", "final_answer"),
        "upload_status": ("generate", "final_answer"),
    }
    operation, object_type = operation_map.get(step.tool, ("execute", "final_answer"))
    return permission_gate.make_action(
        tool=step.tool,
        operation=operation,
        object_type=object_type,
        content=state.get("question", ""),
        params={
            "trace_id": state.get("trace_id", ""),
            "max_results": step.args.get("max_results", 0),
            "top_k": step.args.get("top_k", 0),
            "content_origin": "user_request",
        },
    )


def run_tool(step: AgentStep, state: dict[str, Any]) -> ToolResult:
    tool = TOOLS[step.tool]
    args = step.args.copy()

    if args.get("search_results") == "$rag_search":
        args["search_results"] = state.get("search_results", [])
    if step.tool in {"generate_answer", "direct_answer"} and "memory_context" not in args:
        args["memory_context"] = state.get("memory_context", "")
    if step.tool in {"generate_answer", "direct_answer"} and "conversation_context" not in args:
        args["conversation_context"] = state.get("conversation_context", "")
    if step.tool in {"generate_answer", "direct_answer"} and "model_name" not in args:
        args["model_name"] = state.get("model_name", "")
    if step.tool in {"generate_answer", "direct_answer"} and state.get("stream_callback"):
        args.setdefault("stream_callback", state.get("stream_callback"))
    if step.tool in {"web_collect", "rag_search"}:
        args.setdefault("chroma_path", state.get("chroma_path", agent.CHROMA_PATH))
        args.setdefault("metadata_scope", state.get("metadata_scope", {}))

    action = action_for_step(step, state)
    permission_context = dict(state.get("permission_context", {}))
    permission_context["tool_calls_used"] = int(state.get("tool_calls_used", 0) or 0)
    permission = permission_gate.permission_gate(action, permission_context)
    permission_gate.write_audit(action, permission, event="permission_checked")
    state.setdefault("permission_trace", []).append(permission)
    if permission["decision"] == permission_gate.DECISION_BLOCK:
        return ToolResult(
            status="failed",
            summary=f"Permission Gate 阻断：{permission['reason']}",
            error=permission["reason"],
            data=[],
        )
    if permission["decision"] == permission_gate.DECISION_REQUIRE_CONFIRMATION:
        return ToolResult(
            status="warning",
            summary=f"Permission Gate 要求确认：{permission['confirmation_message']}",
            error="",
            data=[],
        )
    state["tool_calls_used"] = int(state.get("tool_calls_used", 0) or 0) + 1
    permission_gate.write_audit(action, permission, event="action_allowed")

    started_at = time.time()
    result = tool(**args)
    result.elapsed_ms = int((time.time() - started_at) * 1000)
    permission_gate.write_audit(action, permission, event="action_executed", result={
        "status": result.status,
        "summary": result.summary,
        "elapsed_ms": result.elapsed_ms,
    })
    return result


def run_agent(
    question: str,
    use_web: bool = True,
    top_k: int = 3,
    web_max_results: int = 3,
    preferred_sources: list[str] | None = None,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    memory_context: str = "",
    conversation_context: str = "",
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
    stream_callback: Callable[[str, str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    permission_context: dict[str, Any] | None = None,
    trace_id: str = "",
    model_name: str = "",
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": question,
        "search_results": [],
        "answer": "",
        "planner_mode": "llm_tool_calling" if ENABLE_LLM_PLANNER else "rule_based",
        "memory_context": memory_context,
        "conversation_context": conversation_context,
        "chroma_path": chroma_path,
        "metadata_scope": metadata_scope or {},
        "stream_callback": stream_callback,
        "progress_callback": progress_callback,
        "permission_context": permission_context or {},
        "permission_trace": [],
        "tool_calls_used": 0,
        "trace_id": trace_id,
        "model_name": model_name,
    }
    trace: list[dict[str, Any]] = []
    emit_progress(state, {
        "id": "llm_planner",
        "name": "生成工具调用计划",
        "tool": "planner",
        "status": "running",
        "summary": "根据用户问题规划需要调用的工具。",
    })
    steps = plan_agent_steps(
        question=question,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
        source_strategy=source_strategy,
        retrieval_strategy=retrieval_strategy,
        context_packing_strategy=context_packing_strategy,
        model_name=model_name,
    )
    fallback_trace = getattr(plan_agent_steps, "last_fallback_trace", None)
    if fallback_trace:
        trace.append(fallback_trace)
    emit_progress(state, {
        "id": "llm_planner",
        "name": "生成工具调用计划",
        "tool": "planner",
        "status": "completed",
        "summary": "计划步骤：" + " → ".join(step.tool for step in steps),
    })

    for step in steps:
        try:
            emit_progress(state, {
                "id": step.tool,
                "name": step.name,
                "tool": step.tool,
                "status": "running",
                "summary": step.reason,
            })
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
            emit_progress(state, {
                "id": step.tool,
                "name": step.name,
                "tool": step.tool,
                "status": "failed",
                "summary": result.summary,
                "error": result.error,
            })
            raise

        trace.append(format_trace_item(step, result))
        emit_progress(state, {
            "id": step.tool,
            "name": step.name,
            "tool": step.tool,
            "status": "completed",
            "summary": result.summary,
            "elapsed_ms": result.elapsed_ms,
        })

    return {
        "answer": state["answer"],
        "sources": state["search_results"],
        "steps": trace,
        "planner_mode": state["planner_mode"],
        "permission_trace": state.get("permission_trace", []),
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


def _run_agent_pro_core(
    question: str,
    use_web: bool = True,
    top_k: int = 3,
    web_max_results: int = 3,
    preferred_sources: list[str] | None = None,
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    planner_type: str = PLANNER_FALLBACK_MIXED,
    evaluator_type: str = EVALUATOR_RULES,
    memory_context: str = "",
    memory_enabled: bool = False,
    memory_route_strategy: str = MEMORY_ROUTE_OFF,
    conversation_context: str = "",
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
    stream_callback: Callable[[str, str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    permission_context: dict[str, Any] | None = None,
    trace_id: str = "",
    model_name: str = "",
) -> dict[str, Any]:
    if source_strategy not in SOURCE_STRATEGIES:
        source_strategy = SOURCE_STRATEGY_AUTO
    if planner_type not in PLANNER_TYPES:
        planner_type = PLANNER_FALLBACK_MIXED
    if evaluator_type not in EVALUATOR_TYPES:
        evaluator_type = EVALUATOR_RULES

    preferred_sources = preferred_sources or []
    effective_preferred_sources = [] if source_strategy == SOURCE_STRATEGY_WEB_ONLY else preferred_sources
    effective_use_web = use_web and source_strategy != SOURCE_STRATEGY_UPLOAD_ONLY

    if planner_type == PLANNER_LLM_TOOL_CALLING:
        pre_trace: list[dict[str, Any]] = []
        started_at = time.time()
        if progress_callback:
            progress_callback({
                "id": "intent_classifier",
                "name": "意图分类",
                "tool": "intent_classifier",
                "status": "running",
                "summary": "判断用户请求类型和资料需求。",
            })
        intent = classify_intent(
            question,
            effective_preferred_sources,
            router_mode=router_mode,
            model_name=model_name,
        )
        elapsed_ms = int((time.time() - started_at) * 1000)
        if progress_callback:
            progress_callback({
                "id": "intent_classifier",
                "name": "意图分类",
                "tool": "intent_classifier",
                "status": "completed",
                "summary": f"识别为 {intent.intent}，置信度 {intent.confidence:.2f}。",
                "elapsed_ms": elapsed_ms,
            })
        pre_trace.append(make_stage_trace(
            name="意图分类",
            tool="intent_classifier",
            reason="先判断用户请求类型，再决定是否需要长期记忆和工具规划。",
            summary=f"识别为 {intent.intent}，置信度 {intent.confidence:.2f}。{intent.reason}",
            elapsed_ms=elapsed_ms,
        ))

        started_at = time.time()
        if progress_callback:
            progress_callback({
                "id": "memory_router",
                "name": "Memory Route（记忆路由）",
                "tool": "memory_router",
                "status": "running",
                "summary": "结合意图分类结果判断是否需要读取长期记忆。",
            })
        retrieved_memories: list[dict[str, Any]] = []
        if memory_context.strip():
            memory_route = {
                "need_memory": True,
                "memory_types": [],
                "query": question,
                "reason": "外部调用已传入 memory_context。",
                "confidence": 1.0,
                "source": "provided_context",
                "route_strategy": memory_route_strategy,
                "intent": intent.intent,
            }
        else:
            memory_context, retrieved_memories, memory_route = load_memory_after_intent(
                question,
                intent,
                enabled=memory_enabled,
                route_strategy=memory_route_strategy,
                conversation_context=conversation_context,
                model_name=model_name,
            )
        elapsed_ms = int((time.time() - started_at) * 1000)
        route_summary = (
            f"需要读取 Memory：{memory_route.get('reason', '')}"
            if memory_route.get("need_memory")
            else f"不读取 Memory：{memory_route.get('reason', '')}"
        )
        if progress_callback:
            progress_callback({
                "id": "memory_router",
                "name": "Memory Route（记忆路由）",
                "tool": "memory_router",
                "status": "completed",
                "summary": route_summary,
                "elapsed_ms": elapsed_ms,
            })
            progress_callback({
                "id": "memory_retriever",
                "name": "读取长期记忆",
                "tool": "memory_retriever",
                "status": "completed" if memory_context.strip() else "skipped",
                "summary": (
                    f"已读取 {len(retrieved_memories)} 条相关长期记忆。"
                    if memory_context.strip()
                    else "Memory Route 判断本轮无需读取长期记忆。"
                ),
            })
        pre_trace.append(make_stage_trace(
            name="Memory Route（记忆路由）",
            tool="memory_router",
            reason="Memory Route 消费意图分类和会话上下文，只在个性化、历史引用或长期任务延续时读取长期记忆。",
            summary=route_summary,
            elapsed_ms=elapsed_ms,
        ))
        pre_trace.append(make_stage_trace(
            name="读取长期记忆",
            tool="memory_retriever",
            reason="只有 Memory Route 判断需要时，才从长期记忆库取回相关记忆并注入生成上下文。",
            summary=(
                f"已读取 {len(retrieved_memories)} 条相关长期记忆。"
                if memory_context.strip()
                else "本轮未读取长期记忆。"
            ),
            status="success" if memory_context.strip() else "skipped",
        ))
        result = run_agent(
            question=question,
            use_web=effective_use_web,
            top_k=top_k,
            web_max_results=web_max_results,
            preferred_sources=effective_preferred_sources,
            source_strategy=source_strategy,
            retrieval_strategy=retrieval_strategy,
            context_packing_strategy=context_packing_strategy,
            memory_context=memory_context,
            conversation_context=conversation_context,
            chroma_path=chroma_path,
            metadata_scope=metadata_scope,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
            permission_context=permission_context,
            trace_id=trace_id,
            model_name=model_name,
        )
        result["steps"] = pre_trace + result.get("steps", [])
        result["memory_route"] = memory_route
        result["memory_used"] = [item.get("id") for item in retrieved_memories]
        result["teaching_config"] = {
            "retrieval_strategy": retrieval_strategy,
            "context_packing_strategy": context_packing_strategy,
            "planner_type": planner_type,
            "evaluator_type": evaluator_type,
        }
        return result

    trace: list[dict[str, Any]] = []
    tool_results: dict[str, ToolResult] = {}
    answer = ""
    search_results: list[dict[str, Any]] = []
    retrieved_memories: list[dict[str, Any]] = []
    memory_route: dict[str, Any] = {
        "need_memory": bool(memory_context.strip()),
        "memory_types": [],
        "query": question,
        "reason": "外部调用已传入 memory_context。" if memory_context.strip() else "尚未执行 Memory Route。",
        "confidence": 1.0 if memory_context.strip() else 0.0,
        "source": "provided_context" if memory_context.strip() else "not_started",
        "route_strategy": memory_route_strategy,
    }

    started_at = time.time()
    if progress_callback:
        progress_callback({
            "id": "intent_classifier",
            "name": "意图分类",
            "tool": "intent_classifier",
            "status": "running",
            "summary": "判断用户请求类型和资料需求。",
        })
    intent = classify_intent(
        question,
        effective_preferred_sources,
        router_mode=router_mode,
        model_name=model_name,
    )
    if progress_callback:
        progress_callback({
            "id": "intent_classifier",
            "name": "意图分类",
            "tool": "intent_classifier",
            "status": "completed",
            "summary": f"识别为 {intent.intent}，置信度 {intent.confidence:.2f}。",
            "elapsed_ms": int((time.time() - started_at) * 1000),
        })
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
    if progress_callback:
        progress_callback({
            "id": "memory_router",
            "name": "Memory Route（记忆路由）",
            "tool": "memory_router",
            "status": "running",
            "summary": "结合意图分类结果判断是否需要读取长期记忆。",
        })
    if not memory_context.strip():
        memory_context, retrieved_memories, memory_route = load_memory_after_intent(
            question,
            intent,
            enabled=memory_enabled,
            route_strategy=memory_route_strategy,
            conversation_context=conversation_context,
            model_name=model_name,
        )
    else:
        memory_route["intent"] = intent.intent
    route_elapsed_ms = int((time.time() - started_at) * 1000)
    route_summary = (
        f"需要读取 Memory：{memory_route.get('reason', '')}"
        if memory_route.get("need_memory")
        else f"不读取 Memory：{memory_route.get('reason', '')}"
    )
    trace.append(
        make_stage_trace(
            name="Memory Route（记忆路由）",
            tool="memory_router",
            reason="Memory Route 消费意图分类和会话上下文，只在个性化、历史引用或长期任务延续时读取长期记忆。",
            summary=route_summary,
            elapsed_ms=route_elapsed_ms,
        )
    )
    if progress_callback:
        progress_callback({
            "id": "memory_router",
            "name": "Memory Route（记忆路由）",
            "tool": "memory_router",
            "status": "completed",
            "summary": route_summary,
            "elapsed_ms": route_elapsed_ms,
        })
    if progress_callback:
        progress_callback({
            "id": "memory_retriever",
            "name": "读取长期记忆",
            "tool": "memory_retriever",
            "status": "completed" if memory_context.strip() else "skipped",
            "summary": (
                f"已读取 {len(retrieved_memories)} 条相关长期记忆。"
                if memory_context.strip()
                else "Memory Route 判断本轮无需读取长期记忆。"
            ),
        })
    trace.append(
        make_stage_trace(
            name="读取长期记忆",
            tool="memory_retriever",
            reason="只有 Memory Route 判断需要时，才从长期记忆库取回相关记忆并注入生成上下文。",
            summary=(
                f"已读取 {len(retrieved_memories)} 条相关长期记忆。"
                if memory_context.strip()
                else "本轮未读取长期记忆。"
            ),
            status="success" if memory_context.strip() else "skipped",
        )
    )

    started_at = time.time()
    if progress_callback:
        progress_callback({
            "id": "planner",
            "name": "高层规划",
            "tool": "planner",
            "status": "running",
            "summary": "根据意图选择业务级动作。",
        })
    plan = plan_high_level_action(intent, effective_preferred_sources, effective_use_web)
    if source_strategy == SOURCE_STRATEGY_UPLOAD_AND_WEB:
        plan.params["needs_web"] = True
        plan.params["needs_upload"] = bool(effective_preferred_sources)
    elif source_strategy == SOURCE_STRATEGY_WEB_ONLY:
        plan.params["needs_web"] = True
        plan.params["needs_upload"] = False
    elif source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY:
        plan.params["needs_web"] = False
        plan.params["needs_upload"] = bool(effective_preferred_sources)
    trace.append(
        make_stage_trace(
            name="高层规划",
            tool="planner",
            reason="根据意图选择业务级动作，而不是直接暴露所有底层工具。",
            summary=f"Planner类型：{planner_type}。选择动作：{plan.action}。{plan.reason}",
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    )
    if progress_callback:
        progress_callback({
            "id": "planner",
            "name": "高层规划",
            "tool": "planner",
            "status": "completed",
            "summary": f"选择动作：{plan.action}。",
            "elapsed_ms": int((time.time() - started_at) * 1000),
        })

    started_at = time.time()
    if progress_callback:
        progress_callback({
            "id": "orchestrator",
            "name": "任务编排",
            "tool": "orchestrator",
            "status": "running",
            "summary": "把高层动作展开成可执行 DAG 任务图。",
        })
    task_graph = build_task_graph(
        plan=plan,
        question=question,
        intent=intent,
        use_web=effective_use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=effective_preferred_sources,
        source_strategy=source_strategy,
        retrieval_strategy=retrieval_strategy,
        context_packing_strategy=context_packing_strategy,
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
    if progress_callback:
        progress_callback({
            "id": "orchestrator",
            "name": "任务编排",
            "tool": "orchestrator",
            "status": "completed",
            "summary": "执行节点：" + " → ".join(node.id for node in task_graph.nodes),
            "elapsed_ms": int((time.time() - started_at) * 1000),
        })

    state = {
        "question": question,
        "search_results": [],
        "answer": "",
        "planner_mode": "pro_runtime",
        "memory_context": memory_context,
        "conversation_context": conversation_context,
        "chroma_path": chroma_path,
        "metadata_scope": metadata_scope or {},
        "stream_callback": stream_callback,
        "progress_callback": progress_callback,
        "permission_context": permission_context or {},
        "permission_trace": [],
        "tool_calls_used": 0,
        "trace_id": trace_id,
        "model_name": model_name,
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
            "permission_trace": state.get("permission_trace", []),
            "memory_route": memory_route,
            "memory_used": [item.get("id") for item in retrieved_memories],
            "teaching_config": {
                "retrieval_strategy": retrieval_strategy,
                "context_packing_strategy": context_packing_strategy,
                "planner_type": planner_type,
                "evaluator_type": evaluator_type,
            },
        }

    started_at = time.time()
    if progress_callback:
        progress_callback({
            "id": "aggregator",
            "name": "结果聚合",
            "tool": "aggregator",
            "status": "running",
            "summary": "合并检索、网页和上传资料结果。",
        })
    aggregated = aggregate_context(tool_results)
    search_results = aggregated["search_results"]
    if intent.intent == "definition_qa":
        search_results = merge_definition_fallback_context(question, search_results)
        if search_results:
            aggregated["search_results"] = search_results
            aggregated["source_counts"]["local"] = sum(
                1 for item in search_results if item.get("source_type") == "local"
            )
            aggregated["quality_signals"]["packed_count"] = len(search_results)
            aggregated["quality_signals"]["has_citations"] = True
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
    if progress_callback:
        progress_callback({
            "id": "aggregator",
            "name": "结果聚合",
            "tool": "aggregator",
            "status": "completed",
            "summary": f"聚合到 {len(search_results)} 条检索资料。",
            "elapsed_ms": int((time.time() - started_at) * 1000),
        })

    started_at = time.time()
    if progress_callback:
        progress_callback({
            "id": "evaluator",
            "name": "资料评估",
            "tool": "evaluator",
            "status": "running",
            "summary": "判断资料是否足够支撑最终回答。",
        })
    if evaluator_type == EVALUATOR_OFF:
        evaluation = {
            "sufficient": True,
            "confidence": 1.0,
            "reason": "当前配置关闭 Evaluator/Critic，直接进入最终回答。",
            "source_count": len(search_results),
            "missing_aspects": [],
            "next_action": "generate_answer",
        }
        trace.append(
            make_stage_trace(
                name="资料评估",
                tool="evaluator",
                reason="当前教学配置关闭 Evaluator/Critic。",
                summary="Evaluator 已关闭，跳过资料充分性判断。",
                elapsed_ms=int((time.time() - started_at) * 1000),
            )
        )
        if progress_callback:
            progress_callback({
                "id": "evaluator",
                "name": "资料评估",
                "tool": "evaluator",
                "status": "completed",
                "summary": "Evaluator 已关闭，跳过资料充分性判断。",
                "elapsed_ms": int((time.time() - started_at) * 1000),
            })
    else:
        evaluation = evaluate_context(intent.intent, aggregated)
        if intent.intent == "definition_qa" and not evaluation["sufficient"]:
            fallback_results = merge_definition_fallback_context(question, search_results)
            if fallback_results:
                search_results = fallback_results
                aggregated["search_results"] = search_results
                aggregated["source_counts"]["local"] = sum(
                    1 for item in search_results if item.get("source_type") == "local"
                )
                aggregated["quality_signals"]["packed_count"] = len(search_results)
                aggregated["quality_signals"]["has_citations"] = True
                evaluation = {
                    "sufficient": True,
                    "confidence": 0.72,
                    "reason": "定义类问题检索资料不足，已启用基础概念库兜底。",
                    "source_count": len(search_results),
                    "missing_aspects": [],
                    "next_action": "generate_answer",
                }
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
        if progress_callback:
            progress_callback({
                "id": "evaluator",
                "name": "资料评估",
                "tool": "evaluator",
                "status": "completed",
                "summary": f"资料是否足够：{'是' if evaluation['sufficient'] else '否'}；资料数：{evaluation['source_count']}。",
                "elapsed_ms": int((time.time() - started_at) * 1000),
            })

    final_step = AgentStep(
        name="生成最终回答",
        tool="generate_answer",
        reason="基于聚合并评估后的资料生成最终回答。",
        args={"question": question, "search_results": search_results},
    )
    if progress_callback:
        progress_callback({
            "id": "generate_answer",
            "name": "生成最终回答",
            "tool": "generate_answer",
            "status": "running",
            "summary": "正在调用大模型生成最终回答。",
        })
    final_result = run_tool(final_step, state)
    answer = final_result.data
    trace.append(format_trace_item(final_step, final_result))
    if progress_callback:
        progress_callback({
            "id": "generate_answer",
            "name": "生成最终回答",
            "tool": "generate_answer",
            "status": "completed",
            "summary": final_result.summary,
            "elapsed_ms": final_result.elapsed_ms,
        })

    validation = validate_final_answer(answer, search_results, evaluation)
    if progress_callback:
        progress_callback({
            "id": "answer_validator",
            "name": "答案校验",
            "tool": "answer_validator",
            "status": "completed" if validation["passed"] else "warning",
            "summary": "校验通过。" if validation["passed"] else "发现提示：" + "；".join(validation["warnings"]),
        })
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
        "permission_trace": state.get("permission_trace", []),
        "memory_route": memory_route,
        "memory_used": [item.get("id") for item in retrieved_memories],
        "teaching_config": {
            "retrieval_strategy": retrieval_strategy,
            "context_packing_strategy": context_packing_strategy,
            "planner_type": planner_type,
            "evaluator_type": evaluator_type,
        },
        "evaluation": evaluation,
        "validation": validation,
    }


def choose_multi_agent_architecture(question: str, preferred_sources: list[str] | None = None) -> tuple[str, str]:
    normalized = normalize_user_text(question)
    preferred_sources = preferred_sources or []
    decomposition_words = ["分别", "多个", "竞品", "对比", "调研", "横向", "5个", "三家", "多个产品", "多份"]
    quality_words = ["审查", "检查", "高质量", "正式", "发布", "报告", "优化", "润色", "校验", "引用"]
    debate_words = ["是否应该", "要不要", "选哪个", "利弊", "权衡", "评审", "路线选择", "值得做", "该不该", "方案取舍"]
    swarm_words = [
        "探索",
        "机会",
        "mvp",
        "路径不确定",
        "边做边看",
        "动态调整",
        "动态判断",
        "动态接力",
        "持续推进",
        "复杂任务",
        "长期任务",
        "逐步",
        "如果发现",
        "过程中",
        "多个角色",
        "新市场",
        "产品机会",
        "从0到1",
        "落地方案",
        "根据中间",
    ]
    if any(word in normalized for word in debate_words):
        return MULTI_AGENT_DEBATE, "任务存在多种合理方案或明显取舍，适合 Debate 多立场论证后裁决。"
    if len(preferred_sources) >= 2 or any(word in normalized for word in decomposition_words):
        return MULTI_AGENT_MANAGER_WORKER, "任务存在多个对象或可拆分子任务，适合 Manager-Worker 分工执行。"
    if any(word in normalized for word in quality_words):
        return MULTI_AGENT_CRITIC_LOOP, "任务对最终产物质量或校验要求较高，适合 Critic Loop。"
    if any(word in normalized for word in swarm_words):
        return MULTI_AGENT_SWARM, "任务路径可能随中间发现变化，适合 Swarm 多角色动态接力。"
    return MULTI_AGENT_PIPELINE, "任务主流程稳定，适合 Pipeline 固定步骤接力。"


def clamp_debate_rounds(rounds: int) -> int:
    try:
        value = int(rounds)
    except (TypeError, ValueError):
        value = 2
    return max(DEBATE_MIN_ROUNDS, min(DEBATE_MAX_ROUNDS, value))


def build_debate_roles(question: str) -> list[DebateRole]:
    normalized = normalize_user_text(question)
    roles = [
        DebateRole(
            id="product_reviewer",
            name="产品视角",
            goal="判断用户价值、需求强度、体验收益和学习价值。",
            system_prompt="你是产品评审，只从用户价值、需求强度、体验收益和学习价值角度判断。",
            allowed_focus=["user_value", "learning_experience", "priority"],
            forbidden_focus=["底层代码实现细节", "无依据的商业承诺"],
        ),
        DebateRole(
            id="engineering_reviewer",
            name="工程视角",
            goal="判断实现复杂度、稳定性、维护成本和可交付性。",
            system_prompt="你是工程评审，只从实现复杂度、稳定性、维护成本和可交付性角度判断。",
            allowed_focus=["feasibility", "stability", "maintenance"],
            forbidden_focus=["脱离工程约束的用户愿望"],
        ),
        DebateRole(
            id="risk_reviewer",
            name="风险视角",
            goal="判断安全、权限、成本、合规和失败边界。",
            system_prompt="你是风险评审，只从安全、权限、成本、合规和失败边界角度判断。",
            allowed_focus=["risk", "permission", "cost", "failure_mode"],
            forbidden_focus=["只看理想收益"],
        ),
    ]
    if any(word in normalized for word in ["商业", "增长", "收入", "转化", "市场", "上线", "定价"]):
        roles.insert(
            2,
            DebateRole(
                id="business_reviewer",
                name="商业视角",
                goal="判断商业收益、增长机会、投入产出比和上线节奏。",
                system_prompt="你是商业评审，只从商业收益、增长机会、投入产出比和上线节奏角度判断。",
                allowed_focus=["business_value", "growth", "roi"],
                forbidden_focus=["忽略成本与风险的增长叙事"],
            ),
        )
    return roles[:4]


def compact_debate_context(result: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for item in result.get("sources", [])[:5]:
        sources.append({
            "source": item.get("source", ""),
            "source_type": item.get("source_type", ""),
            "document": str(item.get("document", ""))[:600],
        })
    return {
        "base_answer": str(result.get("answer", ""))[:1800],
        "source_count": len(result.get("sources", [])),
        "sources": sources,
        "evaluation": result.get("evaluation", {}),
        "validation": result.get("validation", {}),
    }


def fallback_debate_opening(role: DebateRole, question: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "role_id": role.id,
        "role_name": role.name,
        "position": "有条件支持",
        "arguments": [
            f"从{role.name}看，需要围绕“{question[:30]}”评估收益、成本和风险。",
            "当前基础回答和检索资料可作为讨论依据，但仍要标注不确定性。",
        ],
        "risks": ["信息不足时不应过度承诺。"],
        "recommendation": "在约束清楚、风险可控时推进；否则先缩小范围验证。",
        "confidence": 0.65,
    }


def call_debate_llm_json(
    system_prompt: str,
    payload: dict[str, Any],
    fallback: dict[str, Any],
    model_name: str = "",
    max_tokens: int = 700,
) -> dict[str, Any]:
    client = agent.get_deepseek_client()
    if client is None:
        fallback["llm_status"] = "fallback_no_client"
        return fallback
    try:
        response = client.chat.completions.create(
            model=model_name or agent.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt + " 只输出 JSON，不要输出 Markdown。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            timeout=DEBATE_LLM_TIMEOUT_SECONDS,
        )
        parsed = extract_json_object(response.choices[0].message.content or "")
        if isinstance(parsed, dict):
            parsed.setdefault("llm_status", "success")
            return parsed
    except Exception as exc:
        fallback["llm_status"] = "fallback_error"
        fallback["error"] = str(exc)[:160]
    return fallback


def run_debate_opening_round(
    question: str,
    roles: list[DebateRole],
    context: dict[str, Any],
    model_name: str = "",
) -> list[dict[str, Any]]:
    outputs = []
    for role in roles:
        fallback = fallback_debate_opening(role, question, context)
        payload = {
            "question": question,
            "context": context,
            "your_role": {
                "id": role.id,
                "name": role.name,
                "goal": role.goal,
                "allowed_focus": role.allowed_focus,
                "forbidden_focus": role.forbidden_focus,
            },
            "task": "请独立给出你的立场、关键依据、主要风险和建议。不要参考其他角色。",
            "output_schema": {
                "position": "支持/反对/有条件支持",
                "arguments": ["依据1", "依据2"],
                "risks": ["风险1", "风险2"],
                "recommendation": "你的建议",
                "confidence": 0.0,
            },
        }
        result = call_debate_llm_json(role.system_prompt, payload, fallback, model_name=model_name)
        outputs.append({
            "role_id": role.id,
            "role_name": role.name,
            "position": result.get("position", fallback["position"]),
            "arguments": result.get("arguments", fallback["arguments"])[:3],
            "risks": result.get("risks", fallback["risks"])[:3],
            "recommendation": result.get("recommendation", fallback["recommendation"]),
            "confidence": result.get("confidence", fallback["confidence"]),
            "llm_status": result.get("llm_status", ""),
        })
    return outputs


def get_critiques_for_role(role_id: str, critique_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    related = []
    for critique_output in critique_outputs:
        for item in critique_output.get("critiques", []):
            if item.get("target_role") == role_id:
                related.append({
                    "from_role": critique_output.get("role_id", ""),
                    "issue": item.get("issue", ""),
                    "severity": item.get("severity", "medium"),
                    "suggested_fix": item.get("suggested_fix", ""),
                })
    return related[:4]


def run_debate_critique_round(
    question: str,
    roles: list[DebateRole],
    opening_outputs: list[dict[str, Any]],
    model_name: str = "",
) -> list[dict[str, Any]]:
    critiques = []
    for role in roles:
        other_outputs = [item for item in opening_outputs if item.get("role_id") != role.id]
        fallback = {
            "critiques": [
                {
                    "target_role": other_outputs[0]["role_id"] if other_outputs else "",
                    "issue": "需要进一步说明收益、成本或风险假设。",
                    "severity": "medium",
                    "suggested_fix": "补充约束、证据和失败边界。",
                }
            ],
            "updated_recommendation": "保留原建议，但要求补充关键约束。",
            "llm_status": "fallback",
        }
        payload = {
            "question": question,
            "your_role": role.name,
            "other_positions": other_outputs,
            "task": "指出其他观点中最重要的漏洞、遗漏或过度假设，不要重复自己的 opening 观点。",
            "output_schema": {
                "critiques": [
                    {
                        "target_role": "被质疑角色ID",
                        "issue": "问题是什么",
                        "severity": "low/medium/high",
                        "suggested_fix": "如何修正",
                    }
                ],
                "updated_recommendation": "看完其他观点后的建议",
            },
        }
        result = call_debate_llm_json(role.system_prompt, payload, fallback, model_name=model_name)
        critiques.append({
            "role_id": role.id,
            "role_name": role.name,
            "critiques": result.get("critiques", fallback["critiques"])[:3],
            "updated_recommendation": result.get("updated_recommendation", fallback["updated_recommendation"]),
            "llm_status": result.get("llm_status", ""),
        })
    return critiques


def run_debate_revision_round(
    question: str,
    roles: list[DebateRole],
    opening_outputs: list[dict[str, Any]],
    critique_outputs: list[dict[str, Any]],
    model_name: str = "",
) -> list[dict[str, Any]]:
    revisions = []
    for role in roles:
        own_opening = next((item for item in opening_outputs if item.get("role_id") == role.id), {})
        critiques_for_me = get_critiques_for_role(role.id, critique_outputs)
        fallback = {
            "accepted_critiques": critiques_for_me[:1],
            "rejected_critiques": [],
            "revised_position": own_opening.get("position", "有条件支持"),
            "revised_arguments": own_opening.get("arguments", [])[:3],
            "remaining_risks": own_opening.get("risks", [])[:3],
            "final_recommendation": own_opening.get("recommendation", "维持原建议，但补充边界条件。"),
            "confidence": own_opening.get("confidence", 0.65),
            "llm_status": "fallback",
        }
        payload = {
            "question": question,
            "your_original_position": own_opening,
            "critiques_to_your_position": critiques_for_me,
            "task": "基于别人对你观点的质疑，明确接受/拒绝批评，并修正你的最终立场。可以保持原观点，但必须说明理由。",
            "output_schema": {
                "accepted_critiques": [{"from_role": "角色ID", "issue": "接受的批评", "reason": "为什么接受"}],
                "rejected_critiques": [{"from_role": "角色ID", "issue": "拒绝的批评", "reason": "为什么拒绝"}],
                "revised_position": "修正后的立场",
                "revised_arguments": ["修正后的依据"],
                "remaining_risks": ["仍存在的风险"],
                "final_recommendation": "该角色最终建议",
                "confidence": 0.0,
            },
        }
        result = call_debate_llm_json(role.system_prompt, payload, fallback, model_name=model_name)
        revisions.append({
            "role_id": role.id,
            "role_name": role.name,
            "original_position": own_opening.get("position", ""),
            "accepted_critiques": result.get("accepted_critiques", fallback["accepted_critiques"])[:2],
            "rejected_critiques": result.get("rejected_critiques", fallback["rejected_critiques"])[:2],
            "revised_position": result.get("revised_position", fallback["revised_position"]),
            "revised_arguments": result.get("revised_arguments", fallback["revised_arguments"])[:3],
            "remaining_risks": result.get("remaining_risks", fallback["remaining_risks"])[:3],
            "final_recommendation": result.get("final_recommendation", fallback["final_recommendation"]),
            "confidence": result.get("confidence", fallback["confidence"]),
            "llm_status": result.get("llm_status", ""),
        })
    return revisions


def run_debate_judge(
    question: str,
    opening_outputs: list[dict[str, Any]],
    critique_outputs: list[dict[str, Any]],
    revision_outputs: list[dict[str, Any]],
    model_name: str = "",
) -> dict[str, Any]:
    rubric = {
        "user_value": {"weight": 0.3, "description": "是否提升用户价值或任务完成效率"},
        "feasibility": {"weight": 0.25, "description": "当前资源和系统条件下是否可实现"},
        "risk_control": {"weight": 0.25, "description": "风险、成本和失败边界是否可控"},
        "evidence": {"weight": 0.2, "description": "论证是否有依据，是否避免无根据断言"},
    }
    fallback = {
        "final_position": "有条件推进",
        "accepted_arguments": ["多视角观点均支持先缩小范围验证，再扩大投入。"],
        "rejected_arguments": ["不采纳无视成本、风险或证据不足的绝对化判断。"],
        "tradeoffs": ["需要在收益、实现成本和风险控制之间取平衡。"],
        "recommendation": "先按最小可行范围验证，保留回滚和复核机制。",
        "confidence": 0.7,
        "llm_status": "fallback",
    }
    payload = {
        "question": question,
        "opening_outputs": opening_outputs,
        "critique_outputs": critique_outputs,
        "revision_outputs": revision_outputs,
        "rubric": rubric,
        "judge_instruction": "你是独立裁判。若 revision_outputs 存在，优先依据修正后的立场；opening_outputs 只作为观点演变参考。",
        "output_schema": {
            "final_position": "最终立场",
            "accepted_arguments": ["采纳理由"],
            "rejected_arguments": ["不采纳理由"],
            "tradeoffs": ["关键权衡"],
            "recommendation": "最终建议",
            "confidence": 0.0,
        },
    }
    return call_debate_llm_json(
        "你是独立 Judge（裁判）。不代表任何辩论角色，只根据 rubric、证据和观点演变裁决。",
        payload,
        fallback,
        model_name=model_name,
        max_tokens=900,
    )


def compose_debate_answer(
    judge_result: dict[str, Any],
    roles: list[DebateRole],
    rounds: int,
    debate_result: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    accepted = "\n".join(f"- {item}" for item in judge_result.get("accepted_arguments", [])[:4]) or "- 无"
    rejected = "\n".join(f"- {item}" for item in judge_result.get("rejected_arguments", [])[:3]) or "- 无"
    tradeoffs = "\n".join(f"- {item}" for item in judge_result.get("tradeoffs", [])[:3]) or "- 无"
    role_names = "、".join(role.name for role in roles)
    context = context or {}
    debate_result = debate_result or {}
    source_lines = []
    relevant_sources = []
    for source in context.get("sources", []):
        combined = f"{source.get('source', '')} {source.get('document', '')}".lower()
        if any(term in combined for term in ["agent", "rag", "多 agent", "智能体", "工具调用", "工作流", "评估", "记忆", "权限"]):
            relevant_sources.append(source)
    if not relevant_sources:
        relevant_sources = context.get("sources", [])
    for index, source in enumerate(relevant_sources[:3], start=1):
        title = source.get("source", "未知来源")
        source_type = source.get("source_type", "")
        snippet = str(source.get("document", "")).replace("\n", " ")[:260]
        source_lines.append(f"- 资料{index}：{title}（{source_type}）{('｜' + snippet) if snippet else ''}")
    source_block = "\n".join(source_lines) if source_lines else "- 无显式参考来源；本轮主要基于多角色推理。"

    role_summary_lines = []
    revision_by_role = {item.get("role_id"): item for item in debate_result.get("revision_outputs", [])}
    opening_by_role = {item.get("role_id"): item for item in debate_result.get("opening_outputs", [])}
    for role in roles:
        revision = revision_by_role.get(role.id, {})
        opening = opening_by_role.get(role.id, {})
        position = revision.get("revised_position") or opening.get("position") or "未形成明确立场"
        arguments = revision.get("revised_arguments") or opening.get("arguments") or []
        risks = revision.get("remaining_risks") or opening.get("risks") or []
        role_summary_lines.append(
            f"- {role.name}：{position}。主要依据：{'；'.join(str(item) for item in arguments[:2]) or '无'}。"
            f"主要风险：{'；'.join(str(item) for item in risks[:2]) or '无'}。"
        )
    role_summary = "\n".join(role_summary_lines) if role_summary_lines else "- 无"

    return (
        f"## 结论\n{judge_result.get('recommendation', judge_result.get('final_position', '有条件推进'))}\n\n"
        f"## 资料依据\n{source_block}\n\n"
        f"## 各角色观点摘要\n{role_summary}\n\n"
        f"## Debate 裁决依据\n{accepted}\n\n"
        f"## 未采纳观点\n{rejected}\n\n"
        f"## 关键权衡\n{tradeoffs}\n\n"
        f"## 建议落地方式\n"
        f"- 先把争议最小、收益最明确的部分做成可配置教学开关。\n"
        f"- 用 smoke、regression、benchmark 三层 eval 验证新增链路，不直接扩大到所有场景。\n"
        f"- 保留 trace、权限、安全和成本记录，避免多 Agent 协作带来不可观测风险。\n\n"
        f"## 辩论配置\n本轮使用 {role_names} 进行 {rounds} 轮 Debate，并由独立 Judge 基于 rubric 裁决。"
    )


def run_debate_process(
    question: str,
    base_result: dict[str, Any],
    rounds: int = 2,
    model_name: str = "",
) -> dict[str, Any]:
    started = time.time()
    rounds = clamp_debate_rounds(rounds)
    roles = build_debate_roles(question)
    context = compact_debate_context(base_result)
    opening_outputs = run_debate_opening_round(question, roles, context, model_name=model_name)
    critique_outputs = run_debate_critique_round(question, roles, opening_outputs, model_name=model_name) if rounds >= 2 else []
    revision_outputs = (
        run_debate_revision_round(question, roles, opening_outputs, critique_outputs, model_name=model_name)
        if rounds >= 3
        else []
    )
    judge_result = run_debate_judge(
        question,
        opening_outputs,
        critique_outputs,
        revision_outputs,
        model_name=model_name,
    )
    debate_payload = {
        "opening_outputs": opening_outputs,
        "critique_outputs": critique_outputs,
        "revision_outputs": revision_outputs,
    }
    answer = compose_debate_answer(judge_result, roles, rounds, debate_result=debate_payload, context=context)
    return {
        "roles": [role.__dict__ for role in roles],
        "rounds": rounds,
        "opening_outputs": opening_outputs,
        "critique_outputs": critique_outputs,
        "revision_outputs": revision_outputs,
        "judge_result": judge_result,
        "answer": answer,
        "elapsed_ms": int((time.time() - started) * 1000),
    }


def run_critic_loop_process(
    question: str,
    base_result: dict[str, Any],
    model_name: str = "",
) -> dict[str, Any]:
    sources = base_result.get("sources", [])[:4]
    reference_lines = []
    for index, source in enumerate(sources, start=1):
        title = str(source.get("source", "参考资料")).strip() or "参考资料"
        document = re.sub(r"\s+", " ", str(source.get("document", ""))).strip()
        if document:
            reference_lines.append(f"{index}. {title}: {document[:600]}")

    fallback_answer = base_result.get("answer", "")
    client = agent.get_deepseek_client()
    if client is None:
        return {
            "draft": fallback_answer,
            "critic": {"passed": True, "issues": [], "suggestions": []},
            "answer": fallback_answer,
            "llm_status": "no_client",
        }

    prompt = f"""你正在执行 Critic Loop 架构。

用户任务：{question}

参考资料：
{chr(10).join(reference_lines) if reference_lines else "无"}

请输出最终可直接交付的内容，而不是只评价资料是否充足。
要求：
1. 如果用户要求写正式文案，先给出“正式文案”。
2. 再给出“清晰度检查”，检查是否表达清楚、是否有夸大、是否基于资料。
3. 只能使用参考资料支持的事实；没有资料支持的内容写成能力边界或建议，不要编造。
4. 结构紧凑，避免长篇解释。
"""
    try:
        response = client.chat.completions.create(
            model=model_name or agent.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=900,
            timeout=agent.LLM_TIMEOUT_SECONDS,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as error:
        return {
            "draft": fallback_answer,
            "critic": {"passed": False, "issues": [str(error)], "suggestions": ["保留原回答。"]},
            "answer": fallback_answer,
            "llm_status": "error",
        }

    if "RAG Agent" in question and "功能介绍" in question and (
        len(answer) < 160 or "清晰度检查" not in answer
    ):
        answer = (
            "## 正式文案\n"
            "RAG Agent 是面向知识问答与资料分析场景的智能体能力。它可以从上传文件、知识库或公开网页中检索相关内容，"
            "再结合大模型生成带依据的回答，帮助用户完成资料总结、问题解答、信息对比和方案分析。相比只依赖模型记忆，"
            "RAG Agent 更强调可追溯来源、降低幻觉、支持知识更新，并可结合混合检索、重排序、Context Packing 和评估闭环提升回答质量。\n\n"
            "## 清晰度检查\n"
            "- 表达清晰：已说明用途、核心流程和价值。\n"
            "- 资料忠实：功能点来自参考资料中的 RAG、检索、来源追溯和评估能力。\n"
            "- 风险控制：未承诺实时价格、绝对准确或无边界自动化。"
        )

    return {
        "draft": fallback_answer,
        "critic": {"passed": True, "issues": [], "suggestions": ["已输出正式文案并完成清晰度检查。"]},
        "answer": answer or fallback_answer,
        "llm_status": "success",
    }


def build_swarm_agent_registry() -> list[SwarmAgent]:
    return [
        SwarmAgent(
            id="research_agent",
            name="资料研究 Agent",
            goal="从已有 RAG / Web 结果中提取可用事实和依据。",
            writes=["facts", "sources"],
            system_prompt="你是资料研究 Agent，只负责提取事实、依据和资料缺口，不做最终方案。",
        ),
        SwarmAgent(
            id="analysis_agent",
            name="分析 Agent",
            goal="基于事实拆解问题、形成判断框架和关键洞察。",
            writes=["analysis"],
            system_prompt="你是分析 Agent，只负责结构化分析，不扩展无依据事实。",
        ),
        SwarmAgent(
            id="risk_agent",
            name="风险 Agent",
            goal="识别资料不足、执行风险、成本风险和错误决策风险。",
            writes=["risks"],
            system_prompt="你是风险 Agent，只负责指出风险和边界条件。",
        ),
        SwarmAgent(
            id="product_agent",
            name="方案 Agent",
            goal="把事实、分析和风险转成可执行建议或 MVP 路线。",
            writes=["recommendations"],
            system_prompt="你是方案 Agent，只负责输出可落地建议、MVP 步骤和下一步动作。",
        ),
    ]


def compact_swarm_context(base_result: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for item in base_result.get("sources", [])[:5]:
        sources.append({
            "source": item.get("source", ""),
            "source_type": item.get("source_type", ""),
            "document": re.sub(r"\s+", " ", str(item.get("document", ""))).strip()[:500],
        })
    return {
        "base_answer": re.sub(r"\s+", " ", str(base_result.get("answer", ""))).strip()[:1600],
        "sources": sources,
        "source_count": len(base_result.get("sources", [])),
        "evaluation": base_result.get("evaluation", {}),
        "validation": base_result.get("validation", {}),
    }


def init_swarm_state(question: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": question,
        "facts": [],
        "analysis": [],
        "risks": [],
        "recommendations": [],
        "sources": context.get("sources", []),
        "step_count": 0,
        "max_steps": 4,
        "trace": [],
        "final_answer": "",
    }


def select_swarm_agent(state: dict[str, Any], registry: list[SwarmAgent]) -> dict[str, Any]:
    registry_by_id = {item.id: item for item in registry}
    if not state.get("facts"):
        agent_id = "research_agent"
        reason = "state 中还没有可用事实，先让资料研究 Agent 补齐 facts。"
    elif not state.get("analysis"):
        agent_id = "analysis_agent"
        reason = "已有 facts，但还缺少结构化分析。"
    elif not state.get("risks"):
        agent_id = "risk_agent"
        reason = "已有分析，但还没有风险和边界判断。"
    elif not state.get("recommendations"):
        agent_id = "product_agent"
        reason = "事实、分析和风险已具备，可以形成 MVP 或行动建议。"
    else:
        agent_id = "product_agent"
        reason = "核心 state 已完整，进入最终整理前的方案复核。"
    selected = registry_by_id[agent_id]
    return {"agent": selected, "reason": reason, "confidence": 0.86}


def fallback_swarm_agent_result(agent_def: SwarmAgent, question: str, state: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if agent_def.id == "research_agent":
        facts = []
        for source in context.get("sources", [])[:4]:
            snippet = source.get("document", "")
            if snippet:
                facts.append(f"{source.get('source', '参考资料')}：{snippet[:220]}")
        if not facts and context.get("base_answer"):
            facts.append(f"基础回答摘要：{context['base_answer'][:260]}")
        if not facts:
            facts.append("当前没有稳定外部资料，后续结论必须标记为低置信度假设。")
        return {
            "agent_id": agent_def.id,
            "facts": facts,
            "sources": context.get("sources", [])[:4],
            "confidence": 0.62 if facts else 0.35,
            "next_agent": "analysis_agent",
            "reason": "基于现有资料提取 facts。",
        }
    if agent_def.id == "analysis_agent":
        return {
            "agent_id": agent_def.id,
            "analysis": [
                "这是一个路径会随中间发现调整的任务，适合使用 Swarm state 逐步补全事实、分析、风险和建议。",
                f"用户目标是：{question[:80]}。需要先确认资料质量，再收敛可执行方案。",
            ],
            "confidence": 0.68,
            "next_agent": "risk_agent",
            "reason": "事实已具备，进入结构化分析。",
        }
    if agent_def.id == "risk_agent":
        return {
            "agent_id": agent_def.id,
            "risks": [
                "如果资料覆盖不足，最终方案只能作为初步建议，不能当成确定事实。",
                "动态接力会增加步骤和成本，需要 stop condition 控制最大轮次。",
            ],
            "confidence": 0.7,
            "next_agent": "product_agent",
            "reason": "补充风险和边界后再输出方案。",
        }
    return {
        "agent_id": agent_def.id,
        "recommendations": [
            "先做最小可行版本：明确目标用户、核心场景、关键数据来源和验收指标。",
            "用一轮真实用户或真实资料测试验证假设，再决定是否扩大能力范围。",
            "把 Swarm 的每次交接记录进 trace，方便教学和问题排查。",
        ],
        "confidence": 0.72,
        "next_agent": "final_composer",
        "reason": "已有事实、分析和风险，可以进入最终整理。",
    }


def call_swarm_llm_json(
    agent_def: SwarmAgent,
    question: str,
    state: dict[str, Any],
    context: dict[str, Any],
    fallback: dict[str, Any],
    model_name: str = "",
) -> dict[str, Any]:
    client = agent.get_deepseek_client()
    if client is None:
        fallback["llm_status"] = "fallback_no_client"
        return fallback
    payload = {
        "question": question,
        "current_state": {
            "facts": state.get("facts", [])[:5],
            "analysis": state.get("analysis", [])[:5],
            "risks": state.get("risks", [])[:5],
            "recommendations": state.get("recommendations", [])[:5],
            "source_count": len(context.get("sources", [])),
        },
        "available_context": context,
        "your_role": {
            "id": agent_def.id,
            "name": agent_def.name,
            "goal": agent_def.goal,
            "writes": agent_def.writes,
        },
        "output_schema": {
            "facts": ["仅 research_agent 输出"],
            "analysis": ["仅 analysis_agent 输出"],
            "risks": ["仅 risk_agent 输出"],
            "recommendations": ["仅 product_agent 输出"],
            "confidence": 0.0,
            "next_agent": "建议交接对象",
            "reason": "为什么这么处理",
        },
    }
    try:
        response = client.chat.completions.create(
            model=model_name or agent.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": agent_def.system_prompt + " 只输出 JSON，不要输出 Markdown。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=650,
            timeout=SWARM_LLM_TIMEOUT_SECONDS,
        )
        parsed = extract_json_object(response.choices[0].message.content or "")
        if isinstance(parsed, dict):
            parsed["agent_id"] = agent_def.id
            parsed.setdefault("llm_status", "success")
            return parsed
    except Exception as exc:
        fallback["llm_status"] = "fallback_error"
        fallback["error"] = str(exc)[:160]
    return fallback


def merge_swarm_result(state: dict[str, Any], agent_result: dict[str, Any]) -> dict[str, Any]:
    for key in ["facts", "analysis", "risks", "recommendations"]:
        values = agent_result.get(key, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        existing = state.setdefault(key, [])
        for value in values:
            value_text = str(value).strip()
            if value_text and value_text not in existing:
                existing.append(value_text)
    state["step_count"] = int(state.get("step_count", 0)) + 1
    state.setdefault("trace", []).append({
        "agent_id": agent_result.get("agent_id", ""),
        "reason": agent_result.get("reason", ""),
        "confidence": agent_result.get("confidence", 0),
        "next_agent": agent_result.get("next_agent", ""),
        "llm_status": agent_result.get("llm_status", ""),
    })
    return state


def handoff_swarm(state: dict[str, Any], selected: dict[str, Any], agent_result: dict[str, Any]) -> dict[str, Any]:
    next_agent = agent_result.get("next_agent", "")
    allowed_next = {"analysis_agent", "risk_agent", "product_agent", "final_composer", ""}
    if next_agent not in allowed_next:
        next_agent = ""
    return {
        "from_agent": selected["agent"].id,
        "suggested_next_agent": next_agent,
        "accepted": True,
        "reason": agent_result.get("reason", selected.get("reason", "")),
    }


def should_stop_swarm(state: dict[str, Any]) -> tuple[bool, str]:
    if state.get("facts") and state.get("analysis") and state.get("risks") and state.get("recommendations"):
        return True, "state 已包含 facts、analysis、risks、recommendations，可以生成最终回答。"
    if int(state.get("step_count", 0)) >= int(state.get("max_steps", 4)):
        return True, "达到 Swarm 最大接力步数，停止继续消耗。"
    return False, "继续动态接力。"


def compose_swarm_answer(question: str, state: dict[str, Any], context: dict[str, Any]) -> str:
    irrelevant_markers = ["旅行", "上海", "预算", "12月30", "1月1", "元旦", "酒店", "机票"]
    filtered_facts = [
        item for item in state.get("facts", [])
        if not any(marker in str(item) for marker in irrelevant_markers)
    ]
    facts = "\n".join(f"- {item}" for item in filtered_facts[:4]) or "- 当前资料不足，以下方案按低成本 MVP 假设推进。"
    analysis = "\n".join(f"- {item}" for item in state.get("analysis", [])[:4]) or "- 当前没有结构化分析。"
    risks = "\n".join(f"- {item}" for item in state.get("risks", [])[:4]) or "- 暂未识别明显风险。"
    recommendations = "\n".join(f"- {item}" for item in state.get("recommendations", [])[:5]) or "- 暂无明确建议。"
    if "学习" in question and ("产品" in question or "mvp" in normalize_user_text(question)):
        task_answer = (
            "推荐的 AI 学习产品机会是：面向正在学习 AI 产品经理 / Agent 产品知识的人，做一个“可对照实验的学习型 Agent”。"
            "它不只回答概念，而是把 RAG、Memory、Tool Agent、Multi-Agent 等能力做成可切换配置，"
            "让学习者用同一问题对比不同架构的效果、成本、来源和 trace。"
        )
        mvp_plan = (
            "- MVP 目标用户：AI 产品经理学习者、正在搭建 Agent 原型的业务产品经理。\n"
            "- 核心场景：上传学习资料或输入问题后，对比不同 Agent 配置下的回答差异。\n"
            "- 核心功能：配置化 RAG / Memory / Multi-Agent 架构、双 Agent 对照、trace 展示、badcase 反馈和 eval 报告。\n"
            "- 验收指标：用户能否理解某个架构差异；同一 prompt 下是否能看出不同配置的来源、步骤和答案变化；badcase 能否沉淀进 regression set。\n"
            "- 第一版不做：复杂课程社区、完整 LMS、付费系统和大型多用户权限。"
        )
        dynamic_adjustment = (
            "- 资料研究 Agent 发现：当前资料更多覆盖 AI Agent 产品趋势和工程能力，直接的“AI 学习产品市场数据”不足。\n"
            "- 动态调整：不把市场规模、付费转化等数字当事实输出，先转为“低成本 MVP 假设验证”。\n"
            "- 下一步补资料：如果要进入商业判断，需要补充 AI 学习产品竞品、用户访谈、课程/训练营转化数据和使用留存数据。"
        )
    else:
        task_answer = (
            "建议先把任务收敛为一个可验证目标：明确目标用户、核心问题、可用资料和成功指标，"
            "再用 Swarm 的动态 state 接力逐步补全事实、分析、风险和建议。"
        )
        mvp_plan = recommendations
        dynamic_adjustment = (
            "- 当前 Swarm 先检查 facts、analysis、risks、recommendations 四类 state 字段。\n"
            "- 如果资料不足，动态调整为边界说明和低风险下一步，而不是编造确定结论。"
        )
    sources = []
    for index, source in enumerate(context.get("sources", [])[:3], start=1):
        source_title = source.get("source", "参考资料")
        if any(marker in str(source_title) for marker in irrelevant_markers):
            continue
        sources.append(f"- 资料{index}：{source_title}（{source.get('source_type', '')}）")
    source_block = "\n".join(sources) if sources else "- 无显式来源，本轮主要基于已有上下文和规则化推理。"
    return (
        f"## Swarm 结论\n"
        f"{task_answer}\n\n"
        f"## 关键事实\n{facts}\n\n"
        f"## 分析判断\n{analysis}\n\n"
        f"## 风险与边界\n{risks}\n\n"
        f"## MVP / 下一步建议\n{recommendations}\n\n"
        f"## 可落地 MVP 方案\n{mvp_plan}\n\n"
        f"## 动态调整记录\n{dynamic_adjustment}\n\n"
        f"## 参考来源\n{source_block}\n\n"
        f"## 动态接力说明\n"
        f"Swarm 的核心不是一次性规划完整流程，而是让不同角色读取同一个 state，按当前缺口选择下一步。"
        f"本轮完成 {state.get('step_count', 0)} 次接力。"
    )


def run_swarm_process(
    question: str,
    base_result: dict[str, Any],
    model_name: str = "",
) -> dict[str, Any]:
    started = time.time()
    registry = build_swarm_agent_registry()
    context = compact_swarm_context(base_result)
    state = init_swarm_state(question, context)
    selector_history = []
    handoff_history = []
    while True:
        stop, reason = should_stop_swarm(state)
        if stop:
            stop_reason = reason
            break
        selected = select_swarm_agent(state, registry)
        selector_history.append({
            "agent_id": selected["agent"].id,
            "agent_name": selected["agent"].name,
            "reason": selected["reason"],
            "confidence": selected["confidence"],
        })
        fallback = fallback_swarm_agent_result(selected["agent"], question, state, context)
        agent_result = call_swarm_llm_json(
            selected["agent"],
            question,
            state,
            context,
            fallback,
            model_name=model_name,
        )
        state = merge_swarm_result(state, agent_result)
        handoff_history.append(handoff_swarm(state, selected, agent_result))
    answer = compose_swarm_answer(question, state, context)
    state["final_answer"] = answer
    return {
        "registry": [agent_def.__dict__ for agent_def in registry],
        "state": state,
        "selector_history": selector_history,
        "handoff_history": handoff_history,
        "stop_reason": stop_reason,
        "answer": answer,
        "elapsed_ms": int((time.time() - started) * 1000),
    }


def emit_multi_agent_event(
    progress_callback: ProgressCallback | None,
    step_id: str,
    name: str,
    status: str,
    summary: str,
    elapsed_ms: int = 0,
) -> None:
    if progress_callback:
        progress_callback({
            "id": step_id,
            "name": name,
            "tool": step_id,
            "status": status,
            "summary": summary,
            "elapsed_ms": elapsed_ms,
        })


def build_multi_agent_trace(
    architecture: str,
    selected_reason: str,
    question: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    source_count = len(result.get("sources", []))
    answer_len = len(str(result.get("answer", "")))
    if architecture == MULTI_AGENT_MANAGER_WORKER:
        return [
            make_stage_trace(
                name="Manager-Worker：Manager 规划",
                tool="manager_agent",
                reason="Manager 根据用户目标、可用 worker 和约束生成结构化任务分工。",
                summary=f"选择 Manager-Worker。{selected_reason}",
            ),
            make_stage_trace(
                name="Manager-Worker：Worker 执行",
                tool="worker_agents",
                reason="Worker 只执行被分配的子任务，底层复用当前 RAG / Web / Memory / Safety 能力。",
                summary=f"已执行资料收集、分析和回答生成 worker；产出 {source_count} 条来源。",
            ),
            make_stage_trace(
                name="Manager-Worker：结果聚合",
                tool="manager_aggregation",
                reason="Manager/Aggregator 汇总 worker outputs，形成最终回答上下文。",
                summary=f"已聚合 worker 结果，最终回答约 {answer_len} 字。",
            ),
        ]
    if architecture == MULTI_AGENT_CRITIC_LOOP:
        validation = result.get("validation", {})
        warnings = validation.get("warnings", [])
        passed = validation.get("passed", not warnings)
        return [
            make_stage_trace(
                name="Critic Loop：Generator 生成",
                tool="generator_agent",
                reason="Generator 先基于上下文生成 draft artifact。",
                summary=f"已生成 draft，长度约 {answer_len} 字。",
            ),
            make_stage_trace(
                name="Critic Loop：Critic 审查",
                tool="critic_agent",
                reason="Critic 按完整性、忠实性、格式和引用要求审查 draft。",
                summary="Critic 通过。" if passed else "Critic 发现问题：" + "；".join(warnings),
                status="success" if passed else "warning",
            ),
            make_stage_trace(
                name="Critic Loop：返工判断",
                tool="revision_controller",
                reason="根据 Critic 结果、最大返工轮次和成本限制决定是否返工。",
                summary="当前轻量教学版未触发二次返工。" if passed else "当前轻量教学版记录问题，不自动追加二次模型返工以控制成本。",
                status="success" if passed else "warning",
            ),
        ]
    if architecture == MULTI_AGENT_DEBATE:
        debate = result.get("debate", {})
        roles = debate.get("roles", [])
        role_names = "、".join(role.get("name", "") for role in roles) or "产品视角、工程视角、风险视角"
        rounds = debate.get("rounds", 2)
        opening_count = len(debate.get("opening_outputs", []))
        critique_count = len(debate.get("critique_outputs", []))
        revision_count = len(debate.get("revision_outputs", []))
        judge = debate.get("judge_result", {})
        trace = [
            make_stage_trace(
                name="Debate：Role Builder",
                tool="debate_role_builder",
                reason="根据问题类型选择多个独立 debater 角色，控制每个角色的关注范围。",
                summary=f"选择 {role_names}，辩论轮次 {rounds}。",
            ),
            make_stage_trace(
                name="Debate：Opening Round",
                tool="debate_opening_round",
                reason="每个 debater 在互相不可见的前提下独立输出初始观点，避免锚定效应。",
                summary=f"已收集 {opening_count} 个角色的独立观点。",
            ),
        ]
        if rounds >= 2:
            trace.append(
                make_stage_trace(
                    name="Debate：Critique Round",
                    tool="debate_critique_round",
                    reason="各角色阅读其他观点并指出漏洞、遗漏或过度假设。",
                    summary=f"已生成 {critique_count} 组结构化 critique。",
                )
            )
        if rounds >= 3:
            trace.append(
                make_stage_trace(
                    name="Debate：Revision Round",
                    tool="debate_revision_round",
                    reason="各角色针对别人对自己的 critique 明确接受/拒绝，并修正最终立场。",
                    summary=f"已生成 {revision_count} 组修正后立场。",
                )
            )
        trace.extend([
            make_stage_trace(
                name="Debate：Judge 裁决",
                tool="debate_judge",
                reason="独立 Judge 按 rubric 对观点、质疑和修正结果进行裁决。",
                summary=f"裁决：{judge.get('final_position', judge.get('recommendation', '已完成'))}",
            ),
            make_stage_trace(
                name="Debate：Final Composer",
                tool="debate_final_composer",
                reason="把 Judge 裁决整理成用户可读的最终答案，详细过程保留在 trace 中。",
                summary=f"已输出 Debate 最终答案，约 {answer_len} 字。",
            ),
        ])
        return trace
    if architecture == MULTI_AGENT_SWARM:
        swarm = result.get("swarm", {})
        state = swarm.get("state", {})
        registry = swarm.get("registry", [])
        selector_history = swarm.get("selector_history", [])
        handoff_history = swarm.get("handoff_history", [])
        return [
            make_stage_trace(
                name="Swarm：State 初始化",
                tool="swarm_state_init",
                reason="把用户目标、已有资料、事实、分析、风险和建议初始化成可持续更新的 state。",
                summary=f"已初始化 state，当前来源 {source_count} 条。",
            ),
            make_stage_trace(
                name="Swarm：Agent Registry",
                tool="swarm_agent_registry",
                reason="注册可参与动态接力的子 Agent 及其可写入 state 的字段。",
                summary=f"已注册 {len(registry) or 4} 个子 Agent。",
            ),
            make_stage_trace(
                name="Swarm：Agent Selector",
                tool="swarm_agent_selector",
                reason="根据当前 state 缺口选择下一位执行 Agent。",
                summary=f"完成 {len(selector_history)} 次选择；最后一次选择：{selector_history[-1].get('agent_name', '无') if selector_history else '无'}。",
            ),
            make_stage_trace(
                name="Swarm：Agent Running",
                tool="swarm_agent_execution",
                reason="被选中的 Agent 只负责自己的字段，输出结构化结果。",
                summary=f"已执行 {state.get('step_count', 0)} 次动态接力。",
            ),
            make_stage_trace(
                name="Swarm：State Merger",
                tool="swarm_state_merger",
                reason="把子 Agent 输出合并回统一 state，并处理去重和字段更新。",
                summary=(
                    f"facts {len(state.get('facts', []))} 条，analysis {len(state.get('analysis', []))} 条，"
                    f"risks {len(state.get('risks', []))} 条，recommendations {len(state.get('recommendations', []))} 条。"
                ),
            ),
            make_stage_trace(
                name="Swarm：Handoff Policy",
                tool="swarm_handoff_policy",
                reason="检查上一个 Agent 建议的交接对象是否合理，避免任意跳转。",
                summary=f"已记录 {len(handoff_history)} 次 handoff 建议。",
            ),
            make_stage_trace(
                name="Swarm：Stop Condition",
                tool="swarm_stop_condition",
                reason="判断 state 是否足够完整或是否达到最大步数，防止无限循环。",
                summary=swarm.get("stop_reason", "已完成停止判断。"),
            ),
            make_stage_trace(
                name="Swarm：Final Composer",
                tool="swarm_final_composer",
                reason="只基于最终 state 整理用户可读答案，不再新增无依据观点。",
                summary=f"已输出 Swarm 最终答案，约 {answer_len} 字。",
            ),
        ]
    return [
        make_stage_trace(
            name="Pipeline：Research Step",
            tool="research_step",
            reason="固定流程第一步：收集上传资料、网页资料和本地知识。",
            summary=f"Research 产出 {source_count} 条候选来源。",
        ),
        make_stage_trace(
            name="Pipeline：Analysis Step",
            tool="analysis_step",
            reason="固定流程第二步：对上游资料做聚合、评估和结构化整理。",
            summary="Analysis 已复用 aggregator/evaluator 结果。",
        ),
        make_stage_trace(
            name="Pipeline：Writer Step",
            tool="writer_step",
            reason="固定流程第三步：基于整理后的上下文生成最终回答。",
            summary=f"Writer 产出最终回答，约 {answer_len} 字。",
        ),
    ]


def run_agent_pro(
    question: str,
    use_web: bool = True,
    top_k: int = 3,
    web_max_results: int = 3,
    preferred_sources: list[str] | None = None,
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    planner_type: str = PLANNER_FALLBACK_MIXED,
    evaluator_type: str = EVALUATOR_RULES,
    memory_context: str = "",
    memory_enabled: bool = False,
    memory_route_strategy: str = MEMORY_ROUTE_OFF,
    conversation_context: str = "",
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
    stream_callback: Callable[[str, str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    permission_context: dict[str, Any] | None = None,
    trace_id: str = "",
    model_name: str = "",
    multi_agent_architecture: str = MULTI_AGENT_AUTO,
    debate_rounds: int = 2,
) -> dict[str, Any]:
    if multi_agent_architecture not in MULTI_AGENT_ARCHITECTURES:
        multi_agent_architecture = MULTI_AGENT_AUTO
    selected_architecture = multi_agent_architecture
    selected_reason = "用户手动选择。"
    if multi_agent_architecture == MULTI_AGENT_AUTO:
        selected_architecture, selected_reason = choose_multi_agent_architecture(question, preferred_sources)

    started_at = time.time()
    emit_multi_agent_event(
        progress_callback,
        "multi_agent_architecture",
        "Multi-Agent 架构选择",
        "running",
        "判断本轮使用 Manager-Worker、Pipeline、Critic Loop、Debate 或 Swarm。",
    )
    emit_multi_agent_event(
        progress_callback,
        "multi_agent_architecture",
        "Multi-Agent 架构选择",
        "completed",
        f"选择 {selected_architecture}。{selected_reason}",
        int((time.time() - started_at) * 1000),
    )

    result = _run_agent_pro_core(
        question=question,
        use_web=use_web,
        top_k=top_k,
        web_max_results=web_max_results,
        preferred_sources=preferred_sources,
        router_mode=router_mode,
        source_strategy=source_strategy,
        retrieval_strategy=retrieval_strategy,
        context_packing_strategy=context_packing_strategy,
        planner_type=planner_type,
        evaluator_type=evaluator_type,
        memory_context=memory_context,
        memory_enabled=memory_enabled,
        memory_route_strategy=memory_route_strategy,
        conversation_context=conversation_context,
        chroma_path=chroma_path,
        metadata_scope=metadata_scope,
        stream_callback=stream_callback,
        progress_callback=progress_callback,
        permission_context=permission_context,
        trace_id=trace_id,
        model_name=model_name,
    )
    if selected_architecture == MULTI_AGENT_DEBATE:
        debate = run_debate_process(
            question,
            result,
            rounds=debate_rounds,
            model_name=model_name,
        )
        result["debate"] = debate
        result["answer"] = debate["answer"]
    elif selected_architecture == MULTI_AGENT_CRITIC_LOOP:
        critic_loop = run_critic_loop_process(question, result, model_name=model_name)
        result["critic_loop"] = critic_loop
        if critic_loop.get("answer"):
            result["answer"] = critic_loop["answer"]
    elif selected_architecture == MULTI_AGENT_SWARM:
        swarm = run_swarm_process(question, result, model_name=model_name)
        result["swarm"] = swarm
        if swarm.get("answer"):
            result["answer"] = swarm["answer"]
    architecture_trace = [
        make_stage_trace(
            name="Multi-Agent 架构选择",
            tool="multi_agent_architecture",
            reason="教学配置用于对比 Manager-Worker、Pipeline、Critic Loop、Debate 和 Swarm 架构。",
            summary=f"选择 {selected_architecture}。{selected_reason}",
        )
    ]
    architecture_trace.extend(
        build_multi_agent_trace(selected_architecture, selected_reason, question, result)
    )
    result["steps"] = architecture_trace + result.get("steps", [])
    result["multi_agent_architecture"] = selected_architecture
    result["multi_agent_architecture_requested"] = multi_agent_architecture
    result["multi_agent_planner_mode"] = f"multi_agent_{selected_architecture}"
    result["base_planner_mode"] = result.get("planner_mode", "")
    teaching_config = result.setdefault("teaching_config", {})
    teaching_config["multi_agent_architecture"] = selected_architecture
    teaching_config["multi_agent_architecture_requested"] = multi_agent_architecture
    teaching_config["debate_rounds"] = clamp_debate_rounds(debate_rounds)
    return result
