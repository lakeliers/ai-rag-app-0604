import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import rag_agent_core as agent


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

    if len(question.strip()) <= 30 and matches_any_pattern(question, GREETING_PATTERNS):
        return True, "chitchat", "用户输入更像寒暄、自我介绍或普通对话。"

    return False, "", ""


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


def classify_intent_by_llm(question: str, preferred_sources: list[str], rule_result: IntentResult) -> IntentResult:
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
        model=PLANNER_MODEL,
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
) -> IntentResult:
    if router_mode not in ROUTER_MODES:
        router_mode = ROUTER_MODE_RULES

    rule_result = classify_intent_by_rules(question, preferred_sources)
    if router_mode == ROUTER_MODE_RULES or rule_result.confidence >= 0.85:
        rule_result.constraints["router_mode"] = ROUTER_MODE_RULES
        return rule_result

    try:
        llm_result = classify_intent_by_llm(question, preferred_sources, rule_result)
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


def is_definition_fallback_question(question: str) -> bool:
    definition_words = ["是什么", "定义", "概念", "解释一下", "介绍一下", "什么意思", "区别"]
    blocked_words = ["价格", "多少钱", "上线", "日期", "根据这份", "根据我上传", "这份资料", "这个文件"]
    return any(word in question for word in definition_words) and not any(word in question for word in blocked_words)


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


def answer_definition_from_model(question: str) -> str:
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
        model=agent.DEEPSEEK_MODEL,
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
    stream_callback: Callable[[str, str], None] | None = None,
) -> ToolResult:
    generation_error = ""
    if not search_results and is_definition_fallback_question(question):
        try:
            answer = answer_definition_from_model(question)
            agent.conversation_history.append({"role": "user", "content": question})
            agent.conversation_history.append({"role": "assistant", "content": answer})
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
                build_question_with_memory(question, memory_context),
                search_results,
                on_delta=stream_callback,
            )
        else:
            answer = agent.ask_deepseek(build_question_with_memory(question, memory_context), search_results)
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

    agent.conversation_history.append({"role": "user", "content": question})
    agent.conversation_history.append({"role": "assistant", "content": answer})

    return ToolResult(
        status="degraded" if generation_error else "success",
        summary="生成模型请求失败，已使用检索资料兜底回答。" if generation_error else "回答生成完成。",
        data=answer,
        error=generation_error,
    )


def tool_direct_answer(
    question: str,
    memory_context: str = "",
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
        agent.conversation_history.append({"role": "user", "content": question})
        agent.conversation_history.append({"role": "assistant", "content": answer})
        return ToolResult(
            status="success",
            summary="识别为能力介绍问题，已直接说明可用能力。",
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

【长期记忆】
{memory_context or "无"}

用户输入：
{question}
""",
        }
    ]
    if stream_callback:
        chunks = []
        stream = client.chat.completions.create(
            model=agent.DEEPSEEK_MODEL,
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
            model=agent.DEEPSEEK_MODEL,
            messages=messages,
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
    ]
    return matches_any_pattern(normalized, status_patterns)


def run_tool(step: AgentStep, state: dict[str, Any]) -> ToolResult:
    tool = TOOLS[step.tool]
    args = step.args.copy()

    if args.get("search_results") == "$rag_search":
        args["search_results"] = state.get("search_results", [])
    if step.tool in {"generate_answer", "direct_answer"} and "memory_context" not in args:
        args["memory_context"] = state.get("memory_context", "")
    if step.tool in {"generate_answer", "direct_answer"} and state.get("stream_callback"):
        args.setdefault("stream_callback", state.get("stream_callback"))
    if step.tool in {"web_collect", "rag_search"}:
        args.setdefault("chroma_path", state.get("chroma_path", agent.CHROMA_PATH))
        args.setdefault("metadata_scope", state.get("metadata_scope", {}))

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
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    memory_context: str = "",
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
    stream_callback: Callable[[str, str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": question,
        "search_results": [],
        "answer": "",
        "planner_mode": "llm_tool_calling" if ENABLE_LLM_PLANNER else "rule_based",
        "memory_context": memory_context,
        "chroma_path": chroma_path,
        "metadata_scope": metadata_scope or {},
        "stream_callback": stream_callback,
        "progress_callback": progress_callback,
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
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET,
    planner_type: str = PLANNER_FALLBACK_MIXED,
    evaluator_type: str = EVALUATOR_RULES,
    memory_context: str = "",
    chroma_path: str = agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
    stream_callback: Callable[[str, str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if source_strategy not in SOURCE_STRATEGIES:
        source_strategy = SOURCE_STRATEGY_AUTO
    if planner_type not in PLANNER_TYPES:
        planner_type = PLANNER_FALLBACK_MIXED
    if evaluator_type not in EVALUATOR_TYPES:
        evaluator_type = EVALUATOR_RULES

    if planner_type == PLANNER_LLM_TOOL_CALLING:
        result = run_agent(
            question=question,
            use_web=use_web,
            top_k=top_k,
            web_max_results=web_max_results,
            preferred_sources=preferred_sources,
            source_strategy=source_strategy,
            retrieval_strategy=retrieval_strategy,
            context_packing_strategy=context_packing_strategy,
            memory_context=memory_context,
            chroma_path=chroma_path,
            metadata_scope=metadata_scope,
            stream_callback=stream_callback,
            progress_callback=progress_callback,
        )
        result["teaching_config"] = {
            "retrieval_strategy": retrieval_strategy,
            "context_packing_strategy": context_packing_strategy,
            "planner_type": planner_type,
            "evaluator_type": evaluator_type,
        }
        return result

    preferred_sources = preferred_sources or []
    effective_preferred_sources = [] if source_strategy == SOURCE_STRATEGY_WEB_ONLY else preferred_sources
    effective_use_web = use_web and source_strategy != SOURCE_STRATEGY_UPLOAD_ONLY
    trace: list[dict[str, Any]] = []
    tool_results: dict[str, ToolResult] = {}
    answer = ""
    search_results: list[dict[str, Any]] = []

    if memory_context.strip():
        if progress_callback:
            progress_callback({
                "id": "memory_retriever",
                "name": "读取长期记忆",
                "tool": "memory_retriever",
                "status": "completed",
                "summary": "已读取与当前问题相关的长期记忆。",
            })
        trace.append(
            make_stage_trace(
                name="读取长期记忆",
                tool="memory_retriever",
                reason="按用户画像、学习偏好和任务进度读取 Memory，并在生成阶段注入上下文。",
                summary="Memory 已启用，本轮会把相关长期记忆注入最终回答。不要把 Memory 当作外部资料来源引用。",
            )
        )

    started_at = time.time()
    if progress_callback:
        progress_callback({
            "id": "intent_classifier",
            "name": "意图分类",
            "tool": "intent_classifier",
            "status": "running",
            "summary": "判断用户请求类型和资料需求。",
        })
    intent = classify_intent(question, effective_preferred_sources, router_mode=router_mode)
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
        "chroma_path": chroma_path,
        "metadata_scope": metadata_scope or {},
        "stream_callback": stream_callback,
        "progress_callback": progress_callback,
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
        "teaching_config": {
            "retrieval_strategy": retrieval_strategy,
            "context_packing_strategy": context_packing_strategy,
            "planner_type": planner_type,
            "evaluator_type": evaluator_type,
        },
        "evaluation": evaluation,
        "validation": validation,
    }
