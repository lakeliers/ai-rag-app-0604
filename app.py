import os
import inspect
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import streamlit as st


def get_secret(name):
    try:
        value = st.secrets[name]
    except Exception:
        value = os.getenv(name, "")
    return value


deepseek_key = get_secret("DEEPSEEK_API_KEY")
dashscope_key = get_secret("DASHSCOPE_API_KEY")
enable_reranker = get_secret("ENABLE_RERANKER")
reranker_model_name = get_secret("RERANKER_MODEL_NAME")
rerank_limit = get_secret("RERANK_LIMIT")
hf_token = get_secret("HF_TOKEN")
enable_summary_chunks = get_secret("ENABLE_SUMMARY_CHUNKS")
summary_min_chars = get_secret("SUMMARY_MIN_CHARS")
enable_llm_planner = get_secret("ENABLE_LLM_PLANNER")
github_token = get_secret("GITHUB_TOKEN")
github_repo = get_secret("GITHUB_REPO")
seed_teaching_memory = get_secret("SEED_TEACHING_MEMORY")

if deepseek_key:
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
if dashscope_key:
    os.environ["DASHSCOPE_API_KEY"] = dashscope_key
if enable_reranker:
    os.environ["ENABLE_RERANKER"] = enable_reranker
if reranker_model_name:
    os.environ["RERANKER_MODEL_NAME"] = reranker_model_name
if rerank_limit:
    os.environ["RERANK_LIMIT"] = rerank_limit
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
if enable_summary_chunks:
    os.environ["ENABLE_SUMMARY_CHUNKS"] = enable_summary_chunks
if summary_min_chars:
    os.environ["SUMMARY_MIN_CHARS"] = summary_min_chars
if enable_llm_planner:
    os.environ["ENABLE_LLM_PLANNER"] = enable_llm_planner
if github_token:
    os.environ["GITHUB_TOKEN"] = github_token
if github_repo:
    os.environ["GITHUB_REPO"] = github_repo

import rag_agent_core as agent
import parsing_layer
import agent_runtime
import autonomous_agent
import badcase_manager
import memory_manager
import permission_gate
import trace_manager

agent.seed_local_note()
if seed_teaching_memory != "0":
    memory_manager.seed_default_memories_if_empty()


st.set_page_config(
    page_title="agent for train",
    page_icon="🤖",
    layout="wide",
)


def read_upload_as_sections(uploaded_file):
    return parsing_layer.read_upload_as_sections(uploaded_file)

def is_image(uploaded_file):
    return uploaded_file.type.startswith("image/")


def format_chunking_strategy(chunking_strategy):
    if isinstance(chunking_strategy, (list, tuple, set)):
        return ",".join(sorted(str(item) for item in chunking_strategy))
    return str(chunking_strategy)


def format_chunking_labels(chunking_strategy):
    value_to_label = {
        value: label
        for label, value in CHUNKING_STRATEGY_LABELS.items()
    }
    if isinstance(chunking_strategy, (list, tuple, set)):
        return "、".join(value_to_label.get(item, str(item)) for item in chunking_strategy)
    return value_to_label.get(chunking_strategy, str(chunking_strategy))


def file_key(uploaded_file, chunking_strategy):
    file_bytes = uploaded_file.getvalue()
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"{uploaded_file.name}:{len(file_bytes)}:{content_hash}:{format_chunking_strategy(chunking_strategy)}"


def ingest_uploaded_files_for_state(uploaded_files, chunking_strategy, *, state_key_prefix="", session_id=None):
    if not uploaded_files:
        return []

    ingested_sources = []
    session_id = session_id or st.session_state.rag_session_id
    metadata_scope = {"session_id": session_id}
    ingested_key = f"{state_key_prefix}ingested_uploads"
    upload_status_key = f"{state_key_prefix}upload_status"
    upload_processing_key = f"{state_key_prefix}upload_processing_status"
    if ingested_key not in st.session_state:
        st.session_state[ingested_key] = {}
    if upload_status_key not in st.session_state:
        st.session_state[upload_status_key] = []
    if upload_processing_key not in st.session_state:
        st.session_state[upload_processing_key] = {}

    for uploaded_file in uploaded_files:
        key = file_key(uploaded_file, chunking_strategy)
        if key in st.session_state[ingested_key]:
            ingested_sources.append(st.session_state[ingested_key][key])
            st.session_state[upload_processing_key][uploaded_file.name] = "已入库，可提问"
            continue

        st.session_state[upload_processing_key][uploaded_file.name] = "入库中"
        if is_image(uploaded_file):
            if not dashscope_key:
                st.warning(f"{uploaded_file.name} 是图片，需要配置 DASHSCOPE_API_KEY 才能解析。")
                st.session_state[upload_processing_key][uploaded_file.name] = "入库失败：缺少图片解析 Key"
                continue

            summary = agent.describe_image_bytes(
                uploaded_file.getvalue(),
                uploaded_file.type,
                "请提取这张图片中的关键信息，整理成适合知识库检索的文字资料。",
            )
            source = f"图片：{uploaded_file.name}"
            chunk_count = agent.add_text_to_chroma(
                summary,
                source=source,
                source_type="upload",
                url=uploaded_file.name,
                content_type="image",
                chunking_strategy=chunking_strategy,
                metadata_scope=metadata_scope,
            )
        else:
            sections = read_upload_as_sections(uploaded_file)
            source = f"上传：{uploaded_file.name}"
            chunk_count = agent.add_sections_to_chroma(
                sections,
                source=source,
                source_type="upload",
                url=uploaded_file.name,
                chunking_strategy=chunking_strategy,
                metadata_scope=metadata_scope,
            )

        st.session_state[ingested_key][key] = source
        st.session_state[upload_status_key].append(f"{source}：{chunk_count} 块｜切分：{format_chunking_labels(chunking_strategy)}")
        st.session_state[upload_processing_key][uploaded_file.name] = f"已入库，可提问｜{chunk_count} 块"
        ingested_sources.append(source)

    return ingested_sources


def ingest_uploaded_files(uploaded_files, question, chunking_strategy):
    return ingest_uploaded_files_for_state(
        uploaded_files,
        chunking_strategy,
        state_key_prefix="",
        session_id=st.session_state.rag_session_id,
    )


def source_label(source):
    source_type = source.get("source_type", "unknown")
    title = source.get("source", "")

    if source_type == "upload" and title.startswith("图片："):
        return "上传图片｜优先"
    if source_type == "upload":
        return "上传资料｜优先"
    if source_type == "web":
        return "网络资料｜补充"
    if source_type == "local":
        return "基础资料｜兜底"
    return "其他资料｜参考"


SOURCE_STRATEGY_LABELS = {
    "自动判断": "auto",
    "仅上传资料": "upload_only",
    "仅联网资料": "web_only",
    "上传资料 + 联网并行": "upload_and_web",
}
RETRIEVAL_STRATEGY_LABELS = {
    "仅向量检索": "vector_only",
    "向量 + BM25": "vector_bm25",
    "向量 + BM25 + RRF": "vector_bm25_rrf",
}
CONTEXT_PACKING_LABELS = {
    "简单 TopK（取前 K 条资料）": "simple_topk",
    "来源优先": "source_priority",
    "去重 + 新鲜度 + 来源权重": "weighted",
    "严格 token budget（令牌预算）": "strict_budget",
}
CHUNKING_STRATEGY_LABELS = {
    "普通文本切分": "plain",
    "Parent-child（父子关系）": "parent_child",
    "表格专用": "table",
    "摘要 chunk（摘要片段）": "summary",
}
PLANNER_TYPE_LABELS = {
    "规则 Planner（规划器）": "rules",
    "LLM Tool Calling Planner（大模型工具调用规划器）": "llm_tool_calling",
    "fallback 混合 Planner（失败回退混合规划器）": "fallback_mixed",
}
EVALUATOR_TYPE_LABELS = {
    "关闭": "off",
    "规则评估": "rules",
}
MEMORY_WRITE_MODE_LABELS = {
    "关闭写入": "off",
    "手动 + 半自动确认": "confirm",
}
MEMORY_ROUTE_STRATEGY_LABELS = {
    "自动判断（推荐）": "auto",
    "规则 + LLM 判断": "hybrid",
    "总是读取（教学对比）": "always",
    "关闭读取": "off",
}
MULTI_AGENT_ARCHITECTURE_LABELS = {
    "自动选择": "auto",
    "manager-worker": "manager_worker",
    "pipeline": "pipeline",
    "critic loop": "critic_loop",
    "debate": "debate",
    "swarm": "swarm",
}
DEBATE_ROUND_LABELS = {
    "1轮：独立观点 + Judge": 1,
    "2轮：独立观点 + 互评 + Judge": 2,
    "3轮：独立观点 + 互评 + 修正 + Judge": 3,
}

CATEGORY_LABELS = {
    "chitchat": "chitchat（闲聊）",
    "upload_status": "upload_status（上传状态）",
    "source_scope": "source_scope（资料边界）",
    "web_rag": "web_rag（联网检索增强生成）",
    "document_qa": "document_qa（文档问答）",
    "definition": "definition（概念解释）",
    "autonomous": "autonomous（自主任务）",
    "autonomous_fallback": "autonomous_fallback（自主任务回退）",
    "hybrid_rag": "hybrid_rag（上传资料+联网混合检索）",
}
SEVERITY_LABELS = {
    "low": "low（低）",
    "medium": "medium（中）",
    "high": "high（高）",
    "blocker": "blocker（阻断）",
}
MODE_LABELS = {
    "normal": "normal（普通问答）",
    "autonomous": "autonomous（自主任务）",
    "pro_runtime": "pro_runtime（专业运行时）",
    "autonomous_runtime": "autonomous_runtime（自主任务运行时）",
    "autonomous_fallback": "autonomous_fallback（自主任务回退）",
}
TOOL_LABELS = {
    "direct_answer": "direct_answer（直接回答）",
    "upload_status": "upload_status（上传状态检查）",
    "web_collect": "web_collect（网页收集）",
    "rag_search": "rag_search（检索增强搜索）",
    "generate_answer": "generate_answer（生成回答）",
    "answer_validator": "answer_validator（回答校验）",
}
SOURCE_LABELS = {
    "upload": "upload（上传资料）",
    "web": "web（网页资料）",
    "local": "local（本地基础资料）",
}
DEEPSEEK_MODEL_LABELS = {
    "Flash（快速/低成本）": "deepseek-v4-flash",
    "Pro（高质量/较高成本）": "deepseek-v4-pro",
}


def call_with_supported_kwargs(func, *args, **kwargs):
    supported_params = inspect.signature(func).parameters
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_params
    }
    return func(*args, **filtered_kwargs)


def parse_lines_input(value):
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def extract_tools_from_steps(steps):
    tools = []
    for step in steps or []:
        tool = step.get("tool")
        if tool and tool not in tools:
            tools.append(tool)
    return tools


def extract_source_types(sources):
    source_types = []
    for source in sources or []:
        source_type = source.get("source_type", "")
        if source_type and source_type not in source_types:
            source_types.append(source_type)
    return source_types


def build_conversation_context(messages, max_turns=4, max_chars=1600):
    recent = []
    for message in messages[-max_turns * 2:]:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            recent.append(f"用户：{content[:300]}")
        elif role == "assistant":
            recent.append(f"助手：{content[:300]}")
    if not recent:
        return ""
    text = "\n".join(recent)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return (
        "【本轮会话上下文】以下内容只代表当前浏览器会话中的短期对话历史，"
        "用于回答“刚才/前面/我的名字”等连续对话问题；它不是长期记忆，也不是权威资料。\n"
        + text
    )


def route_memory_with_llm(prompt, conversation_context, rule_route):
    if rule_route.get("confidence", 0) >= 0.85:
        return rule_route
    client = agent.get_deepseek_client()
    if client is None:
        fallback = dict(rule_route)
        fallback["reason"] = fallback.get("reason", "") + " LLM Memory Router 未启用：缺少 DEEPSEEK_API_KEY，已使用规则结果。"
        fallback["source"] = "rule_fallback"
        return fallback
    payload = {
        "question": prompt,
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
            model=deepseek_model,
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
        parsed = agent_runtime.extract_json_object(response.choices[0].message.content or "{}")
        memory_types = [
            item for item in parsed.get("memory_types", [])
            if item in memory_manager.MEMORY_TYPES
        ]
        confidence = float(parsed.get("confidence", rule_route.get("confidence", 0.6)))
        return {
            "need_memory": bool(parsed.get("need_memory", rule_route.get("need_memory", False))),
            "memory_types": memory_types,
            "query": str(parsed.get("query") or prompt),
            "reason": f"LLM Memory Router：{parsed.get('reason', '')}",
            "confidence": max(0.0, min(1.0, confidence)),
            "source": "llm",
        }
    except Exception as exc:
        fallback = dict(rule_route)
        fallback["reason"] = fallback.get("reason", "") + f" LLM Memory Router 异常，已使用规则结果：{exc}"
        fallback["source"] = "rule_fallback"
        return fallback


def load_routed_memory(prompt, enabled=True, conversation_context="", route_strategy="auto"):
    if not enabled:
        return "", [], {
            "need_memory": False,
            "memory_types": [],
            "query": prompt,
            "reason": "Memory 开关未启用。",
            "confidence": 1.0,
            "source": "config",
        }
    if route_strategy == "off":
        return "", [], {
            "need_memory": False,
            "memory_types": [],
            "query": prompt,
            "reason": "Memory Route 策略设置为关闭读取。",
            "confidence": 1.0,
            "source": "config",
            "route_strategy": route_strategy,
        }
    if route_strategy == "always":
        memories = memory_manager.retrieve_memories(prompt, include_core=True)
        route = {
            "need_memory": True,
            "memory_types": ["user_profile", "user_preference", "task_progress", "episodic_event", "semantic_rule"],
            "query": prompt,
            "reason": "Memory Route 策略设置为总是读取，用于教学对比。",
            "confidence": 1.0,
            "source": "config",
            "route_strategy": route_strategy,
        }
        return memory_manager.build_memory_context(memories), memories, route
    route = memory_manager.route_memory(prompt, conversation_context=conversation_context)
    if route_strategy == "hybrid":
        route = route_memory_with_llm(prompt, conversation_context, route)
    route["route_strategy"] = route_strategy
    if not route.get("need_memory"):
        return "", [], route
    memories = memory_manager.retrieve_memories(
        route.get("query") or prompt,
        memory_types=route.get("memory_types") or None,
        include_core=True,
    )
    return memory_manager.build_memory_context(memories), memories, route


def generate_trace_id():
    return f"trace_{uuid4().hex[:12]}"


def compact_steps_for_log(steps, limit=30):
    compacted = []
    for step in (steps or [])[:limit]:
        compacted.append({
            "name": step.get("name", ""),
            "tool": step.get("tool", ""),
            "status": step.get("status", ""),
            "summary": str(step.get("summary", ""))[:600],
            "elapsed_ms": step.get("elapsed_ms", 0),
            "error": str(step.get("error", ""))[:600],
        })
    return compacted


def compact_sources_for_log(sources, limit=10):
    compacted = []
    for source in (sources or [])[:limit]:
        compacted.append({
            "source": source.get("source", ""),
            "source_type": source.get("source_type", ""),
            "url": source.get("url", ""),
            "final_score": source.get("final_score", 0),
            "rerank_score": source.get("rerank_score", ""),
            "answerability_score": source.get("answerability_score", 0),
            "document_preview": str(source.get("document", ""))[:600],
        })
    return compacted


def build_trace_record_from_run(run, *, panel_id="", status="success", error=""):
    snapshot = run.get("run_snapshot", {}) or {}
    return {
        "trace_id": run.get("trace_id", ""),
        "event": "agent_turn",
        "status": status,
        "panel_id": panel_id,
        "user_input": run.get("user_input", ""),
        "answer_preview": str(run.get("actual_answer", ""))[:1200],
        "planner_mode": run.get("planner_mode", ""),
        "planner_label": run.get("planner_label", ""),
        "elapsed_ms": run.get("elapsed_ms", snapshot.get("elapsed_ms", 0)),
        "config": run.get("config", {}),
        "tools_called": run.get("tools_called", []),
        "sources_used": run.get("sources_used", []),
        "memory_used": run.get("memory_used", []),
        "steps": snapshot.get("steps", compact_steps_for_log(run.get("steps", []))),
        "sources": snapshot.get("sources", compact_sources_for_log(run.get("sources", []))),
        "permission_trace": run.get("permission_trace", []),
        "error": error,
    }


def persist_trace_for_run(run, *, panel_id="", status="success", error=""):
    trace_result = trace_manager.log_trace(
        build_trace_record_from_run(run, panel_id=panel_id, status=status, error=error),
        online=True,
    )
    run["trace_log"] = trace_result
    return trace_result


def source_type_label(source_type):
    labels = {
        "upload": "上传资料",
        "web": "网页资料",
        "local": "本地基础资料",
        "image": "图片资料",
    }
    return labels.get(source_type or "", source_type or "无")


def summarize_source_types(sources):
    counts = {}
    for source in sources or []:
        source_type = source.get("source_type", "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1
    if not counts:
        return "未引用资料"
    return "、".join(f"{source_type_label(source_type)} {count} 条" for source_type, count in counts.items())


def config_snapshot_pills(config):
    if not config:
        return []
    chunks = config.get("chunking_strategy_labels") or config.get("chunking_strategy") or []
    if isinstance(chunks, str):
        chunks = [chunks]
    return [
        str(config.get("run_mode", "")),
        f"多Agent：{config.get('multi_agent_architecture_label', config.get('multi_agent_architecture', ''))}",
        f"资料：{config.get('source_strategy_label', config.get('source_strategy', ''))}",
        f"检索：{config.get('retrieval_strategy_label', config.get('retrieval_strategy', ''))}",
        f"上下文：{config.get('context_packing_label', config.get('context_packing_strategy', ''))}",
        f"切分：{' + '.join(str(item) for item in chunks) if chunks else '默认'}",
        f"模型：{config.get('deepseek_model_label', config.get('deepseek_model', ''))}",
        f"Reranker：{'开' if config.get('reranker_enabled') else '关'}",
    ]


def render_config_snapshot(config, title="本轮配置快照"):
    pills = [item for item in config_snapshot_pills(config) if item and not item.endswith("：")]
    if not pills:
        return
    html = (
        f'<div class="run-config-snapshot"><span class="snapshot-title">{title}</span>'
        + "".join(f'<span class="config-pill">{pill}</span>' for pill in pills)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def build_current_config():
    return {
        "run_mode": run_mode,
        "multi_agent_architecture_label": multi_agent_architecture_label,
        "multi_agent_architecture": multi_agent_architecture,
        "debate_rounds_label": debate_rounds_label,
        "debate_rounds": debate_rounds,
        "router_mode": router_mode,
        "source_strategy": source_strategy,
        "retrieval_strategy": retrieval_strategy,
        "context_packing_strategy": context_packing_strategy,
        "chunking_strategy": chunking_strategy,
        "chunking_strategy_labels": chunking_strategy_labels,
        "deepseek_model": deepseek_model,
        "deepseek_model_label": deepseek_model_label,
        "planner_type": planner_type,
        "evaluator_type": evaluator_type,
        "memory_enabled": memory_enabled,
        "memory_route_strategy_label": memory_route_strategy_label,
        "memory_route_strategy": memory_route_strategy,
        "memory_write_mode": memory_write_mode,
        "streaming_enabled": streaming_enabled,
        "plan_progress_enabled": plan_progress_enabled,
        "reranker_enabled": agent.ENABLE_RERANKER,
        "top_k": top_k,
        "web_max_results": web_max_results,
        "max_autonomous_steps": max_autonomous_steps,
        "safety_mode": safety_mode,
        "confirmation_policy": confirmation_policy,
        "prompt_injection_guard": prompt_injection_guard,
        "max_tool_calls_per_run": max_tool_calls_per_run,
        "max_web_pages_per_run": max_web_pages_per_run,
    }


SAFETY_MODE_LABELS = {
    "教学模式（推荐）": permission_gate.SAFETY_MODE_LEARNING,
    "严格模式": permission_gate.SAFETY_MODE_STRICT,
    "宽松模式": permission_gate.SAFETY_MODE_RELAXED,
}

CONFIRMATION_POLICY_LABELS = {
    "智能确认（推荐）": permission_gate.CONFIRM_POLICY_SMART,
    "中高风险都确认": permission_gate.CONFIRM_POLICY_ALWAYS,
    "尽量少确认": permission_gate.CONFIRM_POLICY_MINIMAL,
}


def current_permission_context(extra: dict | None = None) -> dict:
    context = {
        "safety_mode": safety_mode,
        "confirmation_policy": confirmation_policy,
        "prompt_injection_guard": prompt_injection_guard,
        "max_tool_calls": max_tool_calls_per_run,
        "max_web_pages": max_web_pages_per_run,
        "tool_calls_used": 0,
        "confirmed_actions": [],
    }
    if extra:
        context.update(extra)
    return context


def check_permission(action, *, confirmed=False):
    context = current_permission_context({
        "confirmed_actions": [action["id"]] if confirmed else [],
    })
    permission = permission_gate.permission_gate(action, context)
    permission_gate.write_audit(action, permission, event="permission_checked")
    return permission


def render_permission_trace(permission_trace):
    if not permission_trace:
        st.caption("本轮没有记录 Permission Gate（权限门）判断。")
        return
    for item in permission_trace:
        st.markdown(f"**{item.get('tool')} / {item.get('operation')} → {item.get('object_type')}**")
        st.caption(
            f"decision（决策）：{item.get('decision')}｜risk（风险）：{item.get('risk_level')}｜"
            f"mode（模式）：{item.get('safety_mode')}"
        )
        st.write(item.get("reason", ""))
        if item.get("confirmation_message"):
            st.info(item["confirmation_message"])
        if item.get("signals"):
            st.json(item["signals"], expanded=False)
        st.divider()


def permission_action_for_memory(candidate, operation="write", action_id=None):
    return permission_gate.make_action(
        tool="memory",
        operation=operation,
        object_type="user_memory",
        content=str(candidate.get("value", "")),
        params={"memory_type": candidate.get("type", ""), "memory_key": candidate.get("key", "")},
        action_id=action_id,
        source="streamlit_ui",
    )


def permission_action_for_memory_id(memory_id, operation):
    return permission_gate.make_action(
        tool="memory",
        operation=operation,
        object_type="user_memory",
        content=f"memory_id={memory_id}",
        params={"memory_id": memory_id},
        action_id=f"memory_{operation}_{memory_id}",
        source="streamlit_ui",
    )


def set_badcase_target(run):
    st.session_state.last_agent_run = run
    st.session_state.show_badcase_form = True


def render_sources_panel(sources, trace_level_value="简洁", run=None):
    if not sources:
        tools = set((run or {}).get("tools_called", []))
        if "direct_answer" in tools:
            st.caption("本轮没有引用资料：问题被识别为闲聊、能力介绍或可直接回答的问题，因此未检索上传资料或网页资料。")
        elif "upload_status" in tools:
            st.caption("本轮没有引用资料：这是上传状态检查，只读取当前侧的上传入库状态。")
        else:
            st.caption("本轮没有引用资料：可能是资料不足、检索未命中，或当前策略不允许使用对应资料来源。")
        return

    for index, source in enumerate(sources, start=1):
        title = source["source"]
        url = source.get("url", "")
        source_type = source.get("source_type", "unknown")
        label = source_label(source)
        st.markdown(f"**{index}. {title}**")
        st.markdown(f"`{label}`")
        st.caption(f"类型：{source_type}｜chunk（资料片段）类型：{source.get('chunk_type', 'child')}")
        location_parts = []
        if source.get("section_title"):
            location_parts.append(f"小节：{source['section_title']}")
        if source.get("page"):
            location_parts.append(f"页码：{source['page']}")
        if source.get("sheet"):
            location_parts.append(f"工作表：{source['sheet']}")
        if source.get("row_start"):
            location_parts.append(f"行：{source.get('row_start')}-{source.get('row_end')}")
        if location_parts:
            st.caption("｜".join(location_parts))
        if url:
            st.write(url)
        st.write(str(source.get("document", ""))[:300])
        if trace_level_value == "完整":
            with st.expander("调试分数", expanded=False):
                st.caption(
                    "融合分："
                    f"{source.get('final_score', 0):.4f}｜"
                    f"原始分：{source.get('pre_rerank_score', source.get('final_score', 0)):.4f}｜"
                    f"意图：{source.get('query_intent', 'general')}｜"
                    f"新鲜度：{source.get('freshness_score', 0):.2f}｜"
                    f"答案性：{source.get('answerability_score', 0):.2f}｜"
                    f"Rerank（重排序）：{source.get('rerank_status', '未启用')}｜"
                    f"Rerank（重排序）分：{source.get('rerank_score', '无')}｜"
                    f"向量排名：{source.get('vector_rank', '未召回')}｜"
                    f"关键词排名：{source.get('bm25_rank', '未召回')}｜"
                    f"上下文顺序：{source.get('context_order', index)}"
                )
        st.divider()


def render_trace_panel(run, trace_level_value="简洁"):
    steps = run.get("steps", [])
    if trace_level_value == "隐藏":
        st.caption("Trace（执行轨迹）已隐藏。")
        return
    if not steps:
        st.caption("本轮没有记录执行步骤。")
        return

    st.caption(
        f"Planner（规划器）：{run.get('planner_label', run.get('planner_mode', ''))}｜"
        f"Trace ID（运行追踪编号）：{run.get('trace_id', '')}"
    )
    for index, step in enumerate(steps, start=1):
        status_map = {
            "success": "成功",
            "warning": "提示",
            "failed": "失败",
        }
        status = status_map.get(step.get("status"), step.get("status", ""))
        st.markdown(f"**{index}. {step.get('name', '')}**")
        st.caption(
            f"工具：{step.get('tool', '')}｜状态：{status}｜耗时：{step.get('elapsed_ms', 0)} ms"
        )
        if trace_level_value == "完整":
            st.write(step.get("reason", ""))
            st.write(step.get("summary", ""))
        else:
            st.write(step.get("summary", ""))
        if step.get("error"):
            st.error(step["error"])
        st.divider()


def render_autonomous_panel(run):
    autonomous = run.get("autonomous", {})
    if not autonomous:
        st.caption("本轮未进入 Autonomous Agent（自主智能体）任务模式。")
        return
    if autonomous.get("goal"):
        st.markdown("**目标**")
        st.write(autonomous["goal"])
        st.caption(f"停止原因：{autonomous.get('stop_reason', '')}")

    tasks = autonomous.get("tasks", [])
    if tasks:
        st.markdown("**任务队列**")
        for task in tasks:
            st.write(f"{task.get('id')}｜{task.get('title')}｜{task.get('status')}")
            st.caption(f"依赖：{', '.join(task.get('depends_on', [])) or '无'}｜预期产物：{task.get('expected_output', '')}")

    critic_results = autonomous.get("critic_results", [])
    if critic_results:
        st.markdown("**Critic（批判器）结果**")
        for critic in critic_results:
            status = "通过" if critic.get("passed") else "未通过"
            st.write(f"{critic.get('task_id')}｜{status}｜分数：{critic.get('score')}")
            if critic.get("issues"):
                st.caption("问题：" + "；".join(critic["issues"]))

    reflections = autonomous.get("reflections", [])
    if reflections:
        st.markdown("**Reflect（反思）补救建议**")
        for reflection in reflections:
            st.write(f"{reflection.get('task_id')} → {reflection.get('repair_task_id')}")
            if reflection.get("issues"):
                st.caption("问题：" + "；".join(reflection["issues"]))


def render_assistant_message(content, run=None, key_suffix=""):
    trace_level_fallback = globals().get("trace_level", "简洁")
    left, right = st.columns([0.94, 0.06], vertical_alignment="top")
    with left:
        st.write(content)
        if run and run.get("trace_id"):
            st.caption(f"Trace ID（运行追踪编号）：{run['trace_id']}")
        if run:
            render_config_snapshot(run.get("config", {}))
    if run:
        with right:
            st.button(
                "反馈",
                key=f"badcase_button_{key_suffix}",
                help="反馈 badcase（不良案例）",
                on_click=set_badcase_target,
                args=(run,),
            )
    if run:
        with st.expander("查看执行细节 / 来源 / Safety / 反馈", expanded=False):
            tab_names = ["执行过程", "来源", "Safety", "反馈"]
            if run.get("memory_used") or st.session_state.get("pending_memory_candidates"):
                tab_names.append("Memory")
            if run.get("autonomous"):
                tab_names.append("自主任务")
            tabs = st.tabs(tab_names)
            for tab_name, tab in zip(tab_names, tabs):
                with tab:
                    if tab_name == "执行过程":
                        render_trace_panel(run, run.get("trace_level", trace_level_fallback))
                    elif tab_name == "来源":
                        render_sources_panel(run.get("sources", []), run.get("trace_level", trace_level_fallback), run)
                    elif tab_name == "Safety":
                        render_permission_trace(run.get("permission_trace", []))
                    elif tab_name == "反馈":
                        st.caption("如果这轮回答有问题，点击回答右侧的“反馈”按钮记录 badcase（不良案例）。")
                        if run.get("trace_id"):
                            st.code(run["trace_id"], language="text")
                        trace_log = run.get("trace_log", {})
                        if trace_log.get("online_url"):
                            st.caption("线上 Trace Log（运行日志）已写入：")
                            st.write(trace_log["online_url"])
                        elif trace_log.get("error"):
                            st.warning(f"Trace Log 写入提示：{trace_log['error']}")
                    elif tab_name == "Memory":
                        used = run.get("memory_used", [])
                        if used:
                            st.caption("本轮使用的 Memory ID（记忆编号）：")
                            st.write("、".join(used))
                        else:
                            st.caption("本轮没有使用长期记忆。")
                    elif tab_name == "自主任务":
                        render_autonomous_panel(run)


PLAN_STATUS_LABELS = {
    "pending": "未执行",
    "running": "正在执行",
    "completed": "已完成",
    "failed": "失败",
    "warning": "提示",
    "skipped": "跳过",
}

PLAN_STATUS_ICONS = {
    "pending": "○",
    "running": "●",
    "completed": "✓",
    "failed": "!",
    "warning": "!",
    "skipped": "-",
}


def base_plan_steps(run_mode_value):
    if run_mode_value == "自主任务":
        return [
            {"id": "goal_manager", "name": "目标结构化", "tool": "Goal Manager", "status": "pending", "summary": "把用户请求转成结构化目标。"},
            {"id": "task_queue", "name": "生成任务队列", "tool": "Task Queue", "status": "pending", "summary": "拆解任务和依赖关系。"},
            {"id": "stop_condition", "name": "检查停止条件", "tool": "Stop Condition", "status": "pending", "summary": "判断是否继续执行。"},
            {"id": "task_collect_context", "name": "收集背景资料", "tool": "Observe-Act Loop", "status": "pending", "summary": "收集上传资料和网页资料。"},
            {"id": "task_extract_findings", "name": "提取关键发现", "tool": "Observe-Act Loop", "status": "pending", "summary": "整理关键发现和缺口。"},
            {"id": "task_write_deliverable", "name": "生成最终交付物", "tool": "Observe-Act Loop", "status": "pending", "summary": "生成结构化任务交付。"},
            {"id": "final_answer", "name": "生成最终回答", "tool": "Final Answer", "status": "pending", "summary": "输出给前端展示。"},
        ]
    return [
        {"id": "multi_agent_architecture", "name": "Multi-Agent 架构选择", "tool": "Multi-Agent", "status": "pending", "summary": "选择 manager-worker、pipeline、critic loop、debate、swarm 或自动判断。"},
        {"id": "intent_classifier", "name": "意图分类", "tool": "Intent Classifier", "status": "pending", "summary": "判断问题类型。"},
        {"id": "memory_router", "name": "Memory Route（记忆路由）", "tool": "Memory Router", "status": "pending", "summary": "结合意图判断是否需要读取长期记忆。"},
        {"id": "memory_retriever", "name": "读取长期记忆", "tool": "Memory", "status": "pending", "summary": "仅在 Memory Route 判断需要时读取相关记忆。"},
        {"id": "planner", "name": "高层规划", "tool": "Planner", "status": "pending", "summary": "决定走普通回答、上传资料、联网或混合 RAG。"},
        {"id": "orchestrator", "name": "任务编排", "tool": "Orchestrator", "status": "pending", "summary": "把计划展开成可执行节点。"},
        {"id": "web_collect", "name": "按需联网收集", "tool": "Web Collect", "status": "pending", "summary": "只有需要网页补充时才读取网页正文。"},
        {"id": "rag_search", "name": "检索可用资料", "tool": "RAG Search", "status": "pending", "summary": "检索本轮可用的上传资料、网页资料和本地知识。"},
        {"id": "aggregator", "name": "结果聚合", "tool": "Aggregator", "status": "pending", "summary": "去重、排序并整理候选资料。"},
        {"id": "evaluator", "name": "资料评估", "tool": "Evaluator", "status": "pending", "summary": "判断资料是否足够支撑回答。"},
        {"id": "generate_answer", "name": "生成最终回答", "tool": "Final Answer", "status": "pending", "summary": "调用大模型生成回答。"},
        {"id": "answer_validator", "name": "答案校验", "tool": "Validator", "status": "pending", "summary": "做空答案、引用和资料不足提示校验。"},
    ]


def merge_plan_event(plan_steps, event):
    step_id = event.get("id") or event.get("tool") or event.get("name")
    for step in plan_steps:
        if step["id"] == step_id:
            step.update({
                "status": event.get("status", step["status"]),
                "summary": event.get("summary", step.get("summary", "")),
                "elapsed_ms": event.get("elapsed_ms", step.get("elapsed_ms", 0)),
                "error": event.get("error", ""),
            })
            return
    plan_steps.append({
        "id": step_id,
        "name": event.get("name", step_id),
        "tool": event.get("tool", ""),
        "status": event.get("status", "pending"),
        "summary": event.get("summary", ""),
        "elapsed_ms": event.get("elapsed_ms", 0),
        "error": event.get("error", ""),
    })


def render_plan_progress(plan_placeholder, plan_steps):
    completed = sum(1 for step in plan_steps if step["status"] in {"completed", "skipped"})
    total = len(plan_steps)
    display_title = "Agent 当前环节"
    display_step = next((step for step in plan_steps if step.get("status") == "running"), None)
    if display_step is None:
        display_step = next(
            (step for step in reversed(plan_steps) if step.get("status") in {"failed", "warning"}),
            None,
        )
    if display_step is None:
        display_step = next(
            (step for step in reversed(plan_steps) if step.get("status") in {"completed", "skipped"}),
            None,
        )
        if display_step is not None:
            display_title = "Agent 最近完成"

    rows = []
    for step in plan_steps:
        status = step.get("status", "pending")
        icon = PLAN_STATUS_ICONS.get(status, "○")
        label = PLAN_STATUS_LABELS.get(status, status)
        summary = step.get("summary", "")
        elapsed = step.get("elapsed_ms", 0)
        elapsed_text = f" · {elapsed}ms" if elapsed else ""
        rows.append(f"{icon} **{label}**｜{step['name']}｜{summary}{elapsed_text}")

    with plan_placeholder.container():
        if display_step:
            current_status = display_step.get("status", "pending")
            current_label = PLAN_STATUS_LABELS.get(current_status, current_status)
            current_icon = PLAN_STATUS_ICONS.get(current_status, "○")
            current_summary = display_step.get("summary", "")
            st.markdown(
                f"**{display_title}**  \n"
                f"{current_icon} **{current_label}**｜{display_step['name']}｜{current_summary}"
            )
        else:
            st.markdown("**Agent 当前环节**  \n○ 等待开始")

        with st.expander(f"查看完整 Plan（计划）执行进度：{completed}/{total} 已完成", expanded=False):
            st.markdown("\n\n".join(rows) if rows else "暂无执行步骤。")


def render_badcase_form():
    run = st.session_state.get("last_agent_run")
    if not run or not st.session_state.get("show_badcase_form"):
        return

    with st.expander("反馈 Bad Case（不良案例）", expanded=True):
        st.markdown("**当前问题现场**")
        if run.get("trace_id"):
            st.code(run["trace_id"], language="text")
        st.write("User Prompt（用户问题）：", run["user_input"])
        with st.expander("查看 Agent Answer（智能体回答）", expanded=False):
            st.write(run["actual_answer"])
            st.caption("工具调用：" + (", ".join(run["tools_called"]) or "无"))
            st.caption("资料来源：" + (", ".join(run["sources_used"]) or "无"))

        default_category = "chitchat"
        if "upload_status" in run["tools_called"]:
            default_category = "upload_status"
        elif "web_collect" in run["tools_called"]:
            default_category = "web_rag"
        elif run["config"].get("run_mode") == "自主任务":
            default_category = "autonomous"

        with st.form("badcase_form"):
            st.markdown("**快速反馈**")
            category = st.selectbox(
                "问题类型",
                badcase_manager.CATEGORIES,
                index=badcase_manager.CATEGORIES.index(default_category),
                format_func=lambda value: CATEGORY_LABELS.get(value, value),
            )
            problem_description = st.text_area(
                "问题说明",
                value="",
                placeholder="说明这轮回答哪里错了，例如：能力介绍问题不应该联网检索，也不应该引用无关网页。",
            )
            save_target = st.radio(
                "保存位置",
                badcase_manager.SAVE_TARGETS,
                index=0,
                horizontal=True,
                help="eval 是评估集合；GitHub Issue 是 GitHub 上的问题单，用于开发者确认 badcase（不良案例）。",
            )
            severity = st.radio(
                "严重级别",
                badcase_manager.SEVERITIES,
                index=1,
                horizontal=True,
                format_func=lambda value: SEVERITY_LABELS.get(value, value),
                help="low=轻微体验问题，medium=影响判断但不阻断，high=明显错误，blocker=核心链路不可用。",
            )

            case_id = badcase_manager.generate_case_id(run["user_input"], category)
            suite = ["regression"]
            selected_mode_default = (
                "autonomous"
                if run["config"].get("run_mode") == "自主任务"
                else "normal"
            )
            selected_mode = selected_mode_default
            expected_mode = ""
            expected_tools = []
            forbidden_tools = []
            expected_sources = []
            forbidden_sources = []
            required_phrases_text = ""
            expected_answer_phrases_text = ""
            forbidden_answer_phrases_text = ""
            min_answer_chars = 20
            success_criteria_text = ""
            note = ""

            with st.expander("高级评测字段", expanded=False):
                case_id = st.text_input(
                    "case_id（用例编号）",
                    value=case_id,
                    help="case_id 是 regression set 里的唯一用例编号。",
                )
                suite = st.multiselect(
                    "suite（评估集合）",
                    badcase_manager.SUITES,
                    default=suite,
                    help="suite 表示这个 case 加入哪个测试集合：smoke 冒烟、regression 回归、benchmark 基准。",
                )
                selected_mode = st.radio(
                    "selected_mode（运行模式）",
                    badcase_manager.SELECTED_MODES,
                    index=badcase_manager.SELECTED_MODES.index(selected_mode_default),
                    horizontal=True,
                    format_func=lambda value: MODE_LABELS.get(value, value),
                )
                expected_mode = st.selectbox(
                    "expected_mode（期望运行时）",
                    [""] + badcase_manager.EXPECTED_MODES,
                    index=0,
                    format_func=lambda value: MODE_LABELS.get(value, "不限制"),
                )
                expected_tools = st.multiselect(
                    "expected_tools（期望调用工具）",
                    badcase_manager.TOOLS,
                    format_func=lambda value: TOOL_LABELS.get(value, value),
                )
                forbidden_tools = st.multiselect(
                    "forbidden_tools（禁止调用工具）",
                    badcase_manager.TOOLS,
                    format_func=lambda value: TOOL_LABELS.get(value, value),
                )
                expected_sources = st.multiselect(
                    "expected_sources（期望资料来源）",
                    badcase_manager.SOURCES,
                    format_func=lambda value: SOURCE_LABELS.get(value, value),
                )
                forbidden_sources = st.multiselect(
                    "forbidden_sources（禁止资料来源）",
                    badcase_manager.SOURCES,
                    format_func=lambda value: SOURCE_LABELS.get(value, value),
                )
                required_phrases_text = st.text_input("required_phrases（必须出现词，逗号分隔）")
                expected_answer_phrases_text = st.text_input("expected_answer_phrases（期望回答词，逗号分隔）")
                forbidden_answer_phrases_text = st.text_input("forbidden_answer_phrases（禁止回答词，逗号分隔）")
                min_answer_chars = st.number_input(
                    "min_answer_chars（最少回答字数）",
                    min_value=0,
                    max_value=1000,
                    value=20,
                    step=1,
                )
                success_criteria_text = st.text_area(
                    "success_criteria（成功标准，每行一条）",
                    value="",
                    placeholder="例如：不得引用历史上传资料\n必须直接介绍 Agent 能力",
                )
                note = st.text_area("note（备注，不参与规则评估）", value="")

            submitted = st.form_submit_button("校验并提交")

        if submitted:
            case = {
                "case_id": case_id.strip(),
                "suite": suite,
                "category": category,
                "user_input": run["user_input"],
                "selected_mode": selected_mode,
                "required_phrases": badcase_manager.split_list(required_phrases_text),
                "expected_answer_phrases": badcase_manager.split_list(expected_answer_phrases_text),
                "forbidden_answer_phrases": badcase_manager.split_list(forbidden_answer_phrases_text),
                "min_answer_chars": int(min_answer_chars),
            }
            if expected_mode:
                case["expected_mode"] = expected_mode
            if expected_tools:
                case["expected_tools"] = expected_tools
            if forbidden_tools:
                case["forbidden_tools"] = forbidden_tools
            if expected_sources:
                case["expected_sources"] = expected_sources
            if forbidden_sources:
                case["forbidden_sources"] = forbidden_sources
            success_criteria = parse_lines_input(success_criteria_text)
            if success_criteria:
                case["success_criteria"] = success_criteria

            try:
                badcase_operations = ["save_local"]
                if save_target in {badcase_manager.SAVE_TARGET_GITHUB, badcase_manager.SAVE_TARGET_BOTH}:
                    badcase_operations.append("create_github_issue")
                blocked_permissions = []
                for operation in badcase_operations:
                    action = permission_gate.make_action(
                        tool="badcase",
                        operation=operation,
                        object_type="regression_case",
                        content=f"{run['user_input']}\n{run['actual_answer']}",
                        params={"trace_id": run.get("trace_id", ""), "save_target": save_target},
                        action_id=f"badcase_{operation}_{case_id}",
                        source="streamlit_ui",
                    )
                    permission = check_permission(action, confirmed=True)
                    if permission["decision"] == permission_gate.DECISION_BLOCK:
                        blocked_permissions.append(permission)
                    else:
                        permission_gate.write_audit(action, permission, event="action_allowed")
                if blocked_permissions:
                    for permission in blocked_permissions:
                        st.error(f"Permission Gate 阻断：{permission['reason']}")
                    return
                save_result = badcase_manager.save_case(
                    save_target=save_target,
                    case=case,
                    actual_answer=run["actual_answer"],
                    config=run["config"],
                    tools_called=run["tools_called"],
                    sources_used=run["sources_used"],
                    trace_id=run.get("trace_id", ""),
                    run_snapshot=run.get("run_snapshot", {}),
                    severity=severity,
                    problem_description=problem_description,
                    note=note,
                )
                if save_result["errors"]:
                    for error in save_result["errors"]:
                        st.error(error)
                else:
                    st.success(f"Badcase ID（不良案例编号）：{save_result['badcase_id']}")
                    if save_result.get("trace_id"):
                        st.info(f"Trace ID（运行追踪编号）：{save_result['trace_id']}")
                    if save_result["local_eval_saved"]:
                        st.success("已写入本地 eval_cases.jsonl。")
                    if save_result["local_badcase_saved"]:
                        st.success("已写入 bad_cases.jsonl。")
                    if save_result["github_issue_url"]:
                        st.success("已创建 GitHub Issue（线上问题单）。")
                        st.write(save_result["github_issue_url"])
                    if save_result.get("github_error"):
                        st.warning(f"本地已保存，但 GitHub Issue 创建失败：{save_result['github_error']}")
            except Exception as error:
                st.error(str(error))


def confirm_memory_candidate(candidate):
    action = permission_action_for_memory(
        candidate,
        operation="write",
        action_id=f"memory_write_{candidate.get('candidate_id', '')}",
    )
    permission = check_permission(action, confirmed=True)
    if permission["decision"] == permission_gate.DECISION_BLOCK:
        st.session_state.memory_notice = "Permission Gate 阻断写入：" + permission["reason"]
        return
    result = memory_manager.upsert_memory(candidate)
    if result["ok"]:
        permission_gate.write_audit(action, permission, event="action_executed", result={"status": "success"})
        st.session_state.memory_notice = "已保存为长期记忆。"
        st.session_state.dismissed_memory_candidates.add(candidate.get("candidate_id"))
    else:
        permission_gate.write_audit(action, permission, event="action_executed", result={"status": "failed", "errors": result["errors"]})
        st.session_state.memory_notice = "保存失败：" + "；".join(result["errors"])


def dismiss_memory_candidate(candidate_id):
    st.session_state.dismissed_memory_candidates.add(candidate_id)


def soft_delete_memory_with_permission(memory_id):
    action = permission_action_for_memory_id(memory_id, "soft_delete")
    permission = check_permission(action, confirmed=True)
    if permission["decision"] == permission_gate.DECISION_BLOCK:
        st.session_state.memory_notice = "Permission Gate 阻断软删除：" + permission["reason"]
        return
    ok = memory_manager.delete_memory(memory_id)
    permission_gate.write_audit(action, permission, event="action_executed", result={"status": "success" if ok else "not_found"})
    st.session_state.memory_notice = "已将该记忆标记为不再使用。" if ok else "未找到该记忆。"


def hard_delete_memory_with_permission(memory_id):
    action = permission_action_for_memory_id(memory_id, "hard_delete")
    permission = check_permission(action, confirmed=True)
    if permission["decision"] == permission_gate.DECISION_BLOCK:
        st.session_state.memory_notice = "Permission Gate 阻断硬删除：" + permission["reason"]
        return
    ok = memory_manager.hard_delete_memory(memory_id)
    permission_gate.write_audit(action, permission, event="action_executed", result={"status": "success" if ok else "not_found"})
    st.session_state.memory_notice = "已永久删除该记忆。" if ok else "未找到该记忆。"


def set_prompt_seed(prompt):
    st.session_state.queued_prompt = prompt


def render_memory_confirmation():
    candidates = st.session_state.get("pending_memory_candidates", [])
    visible_candidates = [
        item for item in candidates
        if item.get("candidate_id") not in st.session_state.dismissed_memory_candidates
    ]
    if not visible_candidates:
        return

    with st.expander("Memory（记忆）候选：确认后才会写入长期记忆", expanded=True):
        st.caption("这是半自动确认模式：Agent 只提出候选，不会把普通对话自动写入长期记忆。")
        for index, candidate in enumerate(visible_candidates):
            decision = memory_manager.memory_write_decision(candidate)
            st.markdown(f"**{index + 1}. {memory_manager.TYPE_LABELS.get(candidate['type'], candidate['type'])}**")
            st.write(candidate["value"])
            st.caption(
                f"key：{candidate['key']}｜scope（范围）：{candidate.get('scope', 'global')}｜"
                f"risk（风险）：{candidate.get('risk_level', 'low')}｜"
                f"confidence（置信度）：{candidate['confidence']:.2f}｜source（来源）：{candidate['source']}"
            )
            if decision == memory_manager.WRITE_DECISION_BLOCK:
                st.warning("该候选命中敏感/禁止记忆规则，只能忽略，不能保存。")
            left, right = st.columns([0.18, 0.82])
            with left:
                st.button(
                    "保存",
                    key=f"save_memory_candidate_{candidate['candidate_id']}",
                    on_click=confirm_memory_candidate,
                    args=(candidate,),
                    disabled=decision == memory_manager.WRITE_DECISION_BLOCK,
                )
            with right:
                st.button(
                    "忽略",
                    key=f"dismiss_memory_candidate_{candidate['candidate_id']}",
                    on_click=dismiss_memory_candidate,
                    args=(candidate["candidate_id"],),
                )
            st.divider()


def render_settings_panel():
    global uploaded_files, run_mode, multi_agent_architecture_label, multi_agent_architecture, debate_rounds_label, debate_rounds, router_mode_label, router_mode, max_autonomous_steps, planner_type_label, planner_type, evaluator_type_label, evaluator_type, memory_enabled, memory_route_strategy_label, memory_route_strategy, memory_write_mode_label, memory_write_mode, source_strategy_label, source_strategy, retrieval_strategy_label, retrieval_strategy, context_packing_label, context_packing_strategy, chunking_strategy_labels, chunking_strategy, top_k, web_max_results, plan_progress_enabled, streaming_enabled, trace_level, deepseek_model_label, deepseek_model, safety_mode_label, safety_mode, confirmation_policy_label, confirmation_policy, prompt_injection_guard, max_tool_calls_per_run, max_web_pages_per_run, show_permission_audit
    st.markdown("### 资料")
    uploaded_files = st.file_uploader(
        "上传文件或图片",
        type=[
            "txt",
            "md",
            "log",
            "pdf",
            "docx",
            "csv",
            "xlsx",
            "json",
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        accept_multiple_files=True,
    )

    st.divider()
    st.markdown("### 核心设置")
    run_mode = st.radio(
        "运行模式",
        ["普通问答", "自主任务"],
        horizontal=True,
        help="Agent 是智能体；Tool Agent 是会调用工具完成任务的智能体。",
    )
    source_strategy_label = st.radio(
        "资料来源策略",
        list(SOURCE_STRATEGY_LABELS.keys()),
        horizontal=False,
        help="用于观察上传资料、网页资料和自动策略对结果的影响。",
    )
    source_strategy = SOURCE_STRATEGY_LABELS[source_strategy_label]

    with st.expander("检索与切分设置", expanded=False):
        retrieval_strategy_label = st.selectbox(
            "检索策略",
            list(RETRIEVAL_STRATEGY_LABELS.keys()),
            index=2,
            help="BM25 是关键词检索算法；RRF 是多路召回结果融合排序方法。",
        )
        retrieval_strategy = RETRIEVAL_STRATEGY_LABELS[retrieval_strategy_label]
        context_packing_label = st.selectbox(
            "Context Packing（上下文打包）策略",
            list(CONTEXT_PACKING_LABELS.keys()),
            index=3,
            help="Context Packing 是把候选资料筛选、去重并打包进模型上下文的过程。",
        )
        context_packing_strategy = CONTEXT_PACKING_LABELS[context_packing_label]
        chunking_strategy_labels = st.multiselect(
            "Chunking（切分）策略",
            list(CHUNKING_STRATEGY_LABELS.keys()),
            default=["Parent-child（父子关系）", "表格专用"],
            help=(
                "Chunking 是把文档切成适合检索的小片段；这里是启用哪些切分能力。"
                "后端会根据解析出的内容类型自动路由，摘要 chunk 是附加策略。"
            ),
        )
        if not chunking_strategy_labels:
            st.warning("至少选择一种 Chunking（切分）策略；当前已按 Parent-child（父子关系）处理。")
            chunking_strategy_labels = ["Parent-child（父子关系）"]
        chunking_strategy = [
            CHUNKING_STRATEGY_LABELS[label]
            for label in chunking_strategy_labels
        ]
        top_k = st.slider("资料条数", 1, 5, 3)
        web_max_results = st.slider("网页结果数", 1, 5, 2)
        reranker_enabled = st.toggle(
            "启用 Reranker（重排序器）",
            value=agent.ENABLE_RERANKER,
            help="Reranker 会对初步召回的资料做精排，让更相关的资料排前面。",
        )
        agent.ENABLE_RERANKER = reranker_enabled

    with st.expander("Agent 高级设置", expanded=False):
        multi_agent_architecture_label = st.selectbox(
            "Multi-Agent（多智能体）架构",
            list(MULTI_AGENT_ARCHITECTURE_LABELS.keys()),
            index=0,
            help="用于教学对比 Manager-Worker、Pipeline、Critic Loop、Debate、Swarm 等多智能体架构。",
        )
        multi_agent_architecture = MULTI_AGENT_ARCHITECTURE_LABELS[multi_agent_architecture_label]
        debate_rounds_label = st.selectbox(
            "Debate（辩论）轮次",
            list(DEBATE_ROUND_LABELS.keys()),
            index=1,
            help="只在 Multi-Agent 架构选择 debate 或自动选择命中 debate 时生效；轮次越多，观点修正越充分，但成本和等待时间越高。",
        )
        debate_rounds = DEBATE_ROUND_LABELS[debate_rounds_label]
        router_mode_label = st.radio(
            "路由模式",
            ["规则路由", "规则-LLM-规则路由"],
            help="LLM 是大语言模型；规则-LLM-规则表示先规则兜底，再模型分类，最后规则复核。",
        )
        router_mode = (
            "hybrid"
            if router_mode_label == "规则-LLM-规则路由"
            else "rules"
        )
        max_autonomous_steps = st.slider("自主任务最大步数", 1, 5, 3)
        planner_type_label = st.selectbox(
            "Planner（规划器）类型",
            list(PLANNER_TYPE_LABELS.keys()),
            index=2,
            help="Planner 是规划器；Tool Calling 是工具调用；fallback 是失败后的回退策略。",
        )
        planner_type = PLANNER_TYPE_LABELS[planner_type_label]
        evaluator_type_label = st.selectbox(
            "Evaluator / Critic（评估器 / 批判器）",
            list(EVALUATOR_TYPE_LABELS.keys()),
            index=1,
            help="Evaluator 判断资料是否足够；Critic 检查中间产物或最终回答是否达标。",
        )
        evaluator_type = EVALUATOR_TYPE_LABELS[evaluator_type_label]

    with st.expander("Safety / Permission（安全与权限）", expanded=False):
        safety_mode_label = st.selectbox(
            "Safety Mode（安全模式）",
            list(SAFETY_MODE_LABELS.keys()),
            index=0,
            help="教学模式展示主流权限链路；严格模式会让更多中风险动作需要确认；宽松模式减少确认但仍阻断敏感信息。",
        )
        safety_mode = SAFETY_MODE_LABELS[safety_mode_label]
        confirmation_policy_label = st.selectbox(
            "Human Confirmation（用户确认）策略",
            list(CONFIRMATION_POLICY_LABELS.keys()),
            index=0,
            help="控制中高风险动作是否需要用户确认。真实生产环境通常按风险等级和对象类型动态判断。",
        )
        confirmation_policy = CONFIRMATION_POLICY_LABELS[confirmation_policy_label]
        prompt_injection_guard = st.toggle(
            "启用 Prompt Injection（提示注入）防护",
            value=True,
            help="外部资料、网页内容和用户上传文件只能作为参考资料，不能提升为系统指令或工具调用指令。",
        )
        max_tool_calls_per_run = st.slider(
            "单轮最大工具调用数",
            3,
            20,
            10,
            help="Runtime Guard：限制一次回答中的工具调用次数，防止循环失控和成本异常。",
        )
        max_web_pages_per_run = st.slider(
            "单轮最大网页读取数",
            1,
            8,
            min(web_max_results, 5),
            help="Runtime Guard：限制联网读取网页数量。该值会约束 Permission Gate 对 web_collect 的放行。",
        )
        show_permission_audit = st.toggle(
            "显示 Permission Audit（权限审计）",
            value=True,
            help="展示最近的权限判断、风险等级、是否放行、是否阻断。",
        )
        if show_permission_audit:
            audit_rows = permission_gate.load_audit(limit=12)
            if not audit_rows:
                st.caption("暂无权限审计记录。")
            else:
                for row in reversed(audit_rows[-6:]):
                    st.caption(
                        f"{row.get('event', '')} · {row.get('tool', '')}/{row.get('operation', '')} · "
                        f"{row.get('decision', '')} · {row.get('risk_level', '')}"
                    )

    with st.expander("Memory（记忆）", expanded=False):
        memory_enabled = st.toggle(
            "启用 Memory（长期记忆）",
            value=True,
            help="Memory 会记住用户画像、学习偏好和任务进度；它不同于 RAG 资料库。",
        )
        memory_route_strategy_label = st.selectbox(
            "Memory Route（记忆路由）策略",
            list(MEMORY_ROUTE_STRATEGY_LABELS.keys()),
            index=0,
            help="控制每轮是否检索长期记忆。自动判断是生产主流做法；总是读取只适合教学对比。",
        )
        memory_route_strategy = MEMORY_ROUTE_STRATEGY_LABELS[memory_route_strategy_label]
        memory_write_mode_label = st.selectbox(
            "Memory 写入模式",
            list(MEMORY_WRITE_MODE_LABELS.keys()),
            index=1,
            help="手动 + 半自动确认表示用户说“记住”或出现长期偏好时，先生成候选，确认后才保存。",
        )
        memory_write_mode = MEMORY_WRITE_MODE_LABELS[memory_write_mode_label]
        if st.button("初始化教学 Memory", help="写入一组教学默认记忆，用于体验 User Memory 和 Task Memory。"):
            seeded = memory_manager.seed_default_memories_if_empty()
            st.success("已初始化默认记忆。" if seeded else "已有记忆，未重复初始化。")
        stats = memory_manager.memory_stats()
        st.caption(
            f"active（生效）：{stats['active']}｜superseded（被替代）：{stats['superseded']}｜"
            f"expired（过期）：{stats['expired']}｜deleted（删除）：{stats['deleted']}"
        )
        if st.button("运行 Memory 维护", help="执行过期处理、重复合并和质量分刷新。"):
            maintenance_result = memory_manager.run_maintenance()
            st.success(
                f"维护完成：过期 {maintenance_result['expired']} 条，合并 {maintenance_result['merged']} 条。"
            )
        with st.expander("查看 / 删除 Memory（记忆）"):
            for item in memory_manager.load_memories():
                if item.get("status") != memory_manager.MEMORY_STATUS_ACTIVE:
                    continue
                st.markdown(f"**{memory_manager.TYPE_LABELS.get(item['type'], item['type'])}**")
                st.write(item["value"])
                st.caption(
                    f"key：{item['key']}｜scope：{item.get('scope', 'global')}｜risk：{item.get('risk_level', 'low')}｜"
                    f"confidence（置信度）：{item['confidence']:.2f}｜quality（质量分）：{item.get('quality_score', 0):.2f}｜"
                    f"use_count（使用次数）：{item.get('use_count', 0)}"
                )
                soft_col, hard_col = st.columns(2)
                with soft_col:
                    st.button(
                        "不再使用",
                        key=f"delete_memory_{item['id']}",
                        on_click=soft_delete_memory_with_permission,
                        args=(item["id"],),
                        help="软删除：保留审计记录，默认不再进入回答上下文。",
                    )
                with hard_col:
                    st.button(
                        "永久删除",
                        key=f"hard_delete_memory_{item['id']}",
                        on_click=hard_delete_memory_with_permission,
                        args=(item["id"],),
                        help="硬删除：用于隐私/安全删除请求，会从 memory store 移除原文。",
                    )
                st.divider()
        with st.expander("Memory 审计日志（最近 20 条）"):
            audit_rows = memory_manager.load_audit(limit=20)
            if not audit_rows:
                st.caption("暂无审计日志。")
            for row in reversed(audit_rows):
                st.caption(f"{row.get('event')}｜{row.get('memory_id', '')}｜{row.get('ts')}")
                st.json(row.get("payload", {}), expanded=False)

    with st.expander("模型与可观测性", expanded=False):
        plan_progress_enabled = st.toggle(
            "显示 Plan（计划）执行进度",
            value=True,
            help="在 Agent 运行过程中展示已完成、正在执行、未执行的计划环节。展示的是代码执行状态，不是模型隐藏推理内容。",
        )
        streaming_enabled = st.toggle(
            "启用 Streaming（流式输出）",
            value=True,
            help=(
                "Streaming 会在最终大模型生成回答时边生成边展示；"
                "关闭后会等完整回答生成完再一次性展示。当前主要覆盖普通问答和 RAG 链路。"
            ),
        )
        trace_level = st.radio(
            "Trace（执行轨迹）展示级别",
            ["简洁", "完整", "隐藏"],
            help="Trace 是 Agent 每一步做了什么、调用了什么工具、耗时多少的记录。",
        )
        default_deepseek_model_index = list(DEEPSEEK_MODEL_LABELS.values()).index(agent.DEEPSEEK_MODEL) if agent.DEEPSEEK_MODEL in DEEPSEEK_MODEL_LABELS.values() else 0
        deepseek_model_label = st.selectbox(
            "DeepSeek Model（模型）",
            list(DEEPSEEK_MODEL_LABELS.keys()),
            index=default_deepseek_model_index,
            help="Flash 更快更省；Pro 通常质量更高但成本和耗时更高。两者共用同一个 DeepSeek API key。",
        )
        deepseek_model = DEEPSEEK_MODEL_LABELS[deepseek_model_label]
        agent.DEEPSEEK_MODEL = deepseek_model
        agent_runtime.PLANNER_MODEL = deepseek_model

    st.divider()
    st.caption("状态")
    st.caption(f"DeepSeek：{'已配置' if deepseek_key else '未配置'}｜通义百炼：{'已配置' if dashscope_key else '未配置'}")
    st.caption(f"Reranker：{'已启用' if agent.ENABLE_RERANKER else '未启用'}｜Streaming：{'已启用' if streaming_enabled else '未启用'}")

    if "upload_status" in st.session_state and st.session_state.upload_status:
        with st.expander("已入库资料", expanded=False):
            for item in st.session_state.upload_status[-8:]:
                st.write(item)


def compare_state_key(panel_id, name):
    return f"compare_{panel_id}_{name}"


def ensure_compare_state(panel_id):
    defaults = {
        "messages": [],
        "rag_session_id": f"compare_{panel_id}_{uuid4().hex[:12]}",
        "last_sources": [],
        "ingested_uploads": {},
        "upload_status": [],
        "last_agent_run": None,
        "pending_memory_candidates": [],
        "dismissed_memory_candidates": set(),
        "memory_notice": "",
        "upload_processing_status": {},
    }
    for name, value in defaults.items():
        key = compare_state_key(panel_id, name)
        if key not in st.session_state:
            st.session_state[key] = value


def permission_context_from_config(config, trace_id):
    return {
        "safety_mode": config["safety_mode"],
        "confirmation_policy": config["confirmation_policy"],
        "prompt_injection_guard": config["prompt_injection_guard"],
        "max_tool_calls": config["max_tool_calls_per_run"],
        "max_web_pages": config["max_web_pages_per_run"],
        "tool_calls_used": 0,
        "confirmed_actions": [],
        "trace_id": trace_id,
    }


def render_compare_upload_state(panel_id, selected_files):
    upload_status = st.session_state.get(compare_state_key(panel_id, "upload_status"), [])
    processing = st.session_state.get(compare_state_key(panel_id, "upload_processing_status"), {})
    selected_names = [item.name for item in selected_files or []]
    if selected_names and not processing:
        st.info(f"已选择 {len(selected_names)} 个文件；下一次发送问题时会先解析并入库。")
        return
    if processing:
        with st.expander("上传处理状态", expanded=False):
            for name in selected_names or list(processing.keys()):
                st.write(f"{name}：{processing.get(name, '已选择，待入库')}")
        return
    if upload_status:
        st.caption(f"已入库 {len(upload_status)} 个资料来源，可直接提问。")
    else:
        st.caption("当前侧暂无上传资料。")


def compare_panel_status_text(panel_id):
    upload_status = st.session_state.get(compare_state_key(panel_id, "upload_status"), [])
    session_id = st.session_state.get(compare_state_key(panel_id, "rag_session_id"), "")
    upload_text = f"已入库 {len(upload_status)} 个资料来源" if upload_status else "无上传资料"
    return f"{upload_text}｜独立会话 {session_id[-6:]}"


def render_compare_settings(panel_id, title):
    st.markdown(f"### {title}")
    st.caption(compare_panel_status_text(panel_id))
    uploaded = st.file_uploader(
        "上传文件或图片",
        type=[
            "txt",
            "md",
            "log",
            "pdf",
            "docx",
            "csv",
            "xlsx",
            "json",
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        accept_multiple_files=True,
        key=f"compare_upload_{panel_id}",
    )
    render_compare_upload_state(panel_id, uploaded)
    run_mode_value = st.radio(
        "运行模式",
        ["普通问答", "自主任务"],
        horizontal=True,
        key=f"compare_run_mode_{panel_id}",
    )
    source_strategy_label_value = st.radio(
        "资料来源策略",
        list(SOURCE_STRATEGY_LABELS.keys()),
        horizontal=True,
        key=f"compare_source_strategy_{panel_id}",
    )

    with st.expander("检索 / 切分", expanded=False):
        retrieval_strategy_label_value = st.selectbox(
            "检索策略",
            list(RETRIEVAL_STRATEGY_LABELS.keys()),
            index=2,
            key=f"compare_retrieval_{panel_id}",
        )
        context_packing_label_value = st.selectbox(
            "Context Packing（上下文打包）策略",
            list(CONTEXT_PACKING_LABELS.keys()),
            index=3,
            key=f"compare_context_packing_{panel_id}",
        )
        chunking_strategy_labels_value = st.multiselect(
            "Chunking（切分）策略",
            list(CHUNKING_STRATEGY_LABELS.keys()),
            default=["Parent-child（父子关系）", "表格专用"],
            key=f"compare_chunking_{panel_id}",
        )
        if not chunking_strategy_labels_value:
            chunking_strategy_labels_value = ["Parent-child（父子关系）"]
            st.warning("至少选择一种切分策略；已按 Parent-child 处理。")
        top_k_value = st.slider("资料条数", 1, 5, 3, key=f"compare_top_k_{panel_id}")
        web_max_results_value = st.slider("网页结果数", 1, 5, 2, key=f"compare_web_max_{panel_id}")

    with st.expander("Agent / 模型", expanded=False):
        multi_agent_architecture_label_value = st.selectbox(
            "Multi-Agent（多智能体）架构",
            list(MULTI_AGENT_ARCHITECTURE_LABELS.keys()),
            index=0,
            key=f"compare_multi_agent_{panel_id}",
        )
        debate_rounds_label_value = st.selectbox(
            "Debate（辩论）轮次",
            list(DEBATE_ROUND_LABELS.keys()),
            index=1,
            key=f"compare_debate_rounds_{panel_id}",
            help="只在本侧选择 debate 或自动命中 debate 时生效。",
        )
        router_mode_label_value = st.radio(
            "路由模式",
            ["规则路由", "规则-LLM-规则路由"],
            key=f"compare_router_{panel_id}",
        )
        max_autonomous_steps_value = st.slider(
            "自主任务最大步数",
            1,
            5,
            3,
            key=f"compare_auto_steps_{panel_id}",
        )
        planner_type_label_value = st.selectbox(
            "Planner（规划器）类型",
            list(PLANNER_TYPE_LABELS.keys()),
            index=2,
            key=f"compare_planner_{panel_id}",
        )
        evaluator_type_label_value = st.selectbox(
            "Evaluator / Critic（评估器 / 批判器）",
            list(EVALUATOR_TYPE_LABELS.keys()),
            index=1,
            key=f"compare_evaluator_{panel_id}",
        )
        deepseek_model_label_value = st.selectbox(
            "DeepSeek Model（模型）",
            list(DEEPSEEK_MODEL_LABELS.keys()),
            index=0,
            key=f"compare_model_{panel_id}",
        )

    with st.expander("Memory / Safety / Trace", expanded=False):
        memory_enabled_value = st.toggle(
            "启用 Memory（长期记忆）",
            value=True,
            key=f"compare_memory_enabled_{panel_id}",
        )
        memory_route_strategy_label_value = st.selectbox(
            "Memory Route（记忆路由）策略",
            list(MEMORY_ROUTE_STRATEGY_LABELS.keys()),
            index=0,
            key=f"compare_memory_route_{panel_id}",
        )
        memory_write_mode_label_value = st.selectbox(
            "Memory 写入模式",
            list(MEMORY_WRITE_MODE_LABELS.keys()),
            index=1,
            key=f"compare_memory_write_{panel_id}",
        )
        safety_mode_label_value = st.selectbox(
            "Safety Mode（安全模式）",
            list(SAFETY_MODE_LABELS.keys()),
            index=0,
            key=f"compare_safety_{panel_id}",
        )
        confirmation_policy_label_value = st.selectbox(
            "Human Confirmation（用户确认）策略",
            list(CONFIRMATION_POLICY_LABELS.keys()),
            index=0,
            key=f"compare_confirmation_{panel_id}",
        )
        prompt_injection_guard_value = st.toggle(
            "启用 Prompt Injection（提示注入）防护",
            value=True,
            key=f"compare_prompt_guard_{panel_id}",
        )
        streaming_enabled_value = st.toggle(
            "启用 Streaming（流式输出）",
            value=True,
            key=f"compare_streaming_{panel_id}",
        )
        plan_progress_enabled_value = st.toggle(
            "显示 Plan（计划）执行进度",
            value=True,
            key=f"compare_plan_progress_{panel_id}",
        )
        trace_level_value = st.radio(
            "Trace（执行轨迹）展示级别",
            ["简洁", "完整", "隐藏"],
            key=f"compare_trace_{panel_id}",
        )
        max_tool_calls_per_run_value = st.slider(
            "单轮最大工具调用数",
            3,
            20,
            10,
            key=f"compare_max_tool_calls_{panel_id}",
        )
        max_web_pages_per_run_value = st.slider(
            "单轮最大网页读取数",
            1,
            8,
            min(web_max_results_value, 5),
            key=f"compare_max_web_pages_{panel_id}",
        )

    upload_status = st.session_state.get(compare_state_key(panel_id, "upload_status"), [])
    if upload_status:
        with st.expander("已入库资料", expanded=False):
            for item in upload_status[-6:]:
                st.write(item)

    return {
        "uploaded_files": uploaded,
        "run_mode": run_mode_value,
        "multi_agent_architecture_label": multi_agent_architecture_label_value,
        "multi_agent_architecture": MULTI_AGENT_ARCHITECTURE_LABELS[multi_agent_architecture_label_value],
        "debate_rounds_label": debate_rounds_label_value,
        "debate_rounds": DEBATE_ROUND_LABELS[debate_rounds_label_value],
        "source_strategy_label": source_strategy_label_value,
        "source_strategy": SOURCE_STRATEGY_LABELS[source_strategy_label_value],
        "retrieval_strategy_label": retrieval_strategy_label_value,
        "retrieval_strategy": RETRIEVAL_STRATEGY_LABELS[retrieval_strategy_label_value],
        "context_packing_label": context_packing_label_value,
        "context_packing_strategy": CONTEXT_PACKING_LABELS[context_packing_label_value],
        "chunking_strategy_labels": chunking_strategy_labels_value,
        "chunking_strategy": [CHUNKING_STRATEGY_LABELS[label] for label in chunking_strategy_labels_value],
        "top_k": top_k_value,
        "web_max_results": web_max_results_value,
        "router_mode_label": router_mode_label_value,
        "router_mode": "hybrid" if router_mode_label_value == "规则-LLM-规则路由" else "rules",
        "max_autonomous_steps": max_autonomous_steps_value,
        "planner_type_label": planner_type_label_value,
        "planner_type": PLANNER_TYPE_LABELS[planner_type_label_value],
        "evaluator_type_label": evaluator_type_label_value,
        "evaluator_type": EVALUATOR_TYPE_LABELS[evaluator_type_label_value],
        "deepseek_model_label": deepseek_model_label_value,
        "deepseek_model": DEEPSEEK_MODEL_LABELS[deepseek_model_label_value],
        "memory_enabled": memory_enabled_value,
        "memory_route_strategy_label": memory_route_strategy_label_value,
        "memory_route_strategy": MEMORY_ROUTE_STRATEGY_LABELS[memory_route_strategy_label_value],
        "memory_write_mode_label": memory_write_mode_label_value,
        "memory_write_mode": MEMORY_WRITE_MODE_LABELS[memory_write_mode_label_value],
        "safety_mode_label": safety_mode_label_value,
        "safety_mode": SAFETY_MODE_LABELS[safety_mode_label_value],
        "confirmation_policy_label": confirmation_policy_label_value,
        "confirmation_policy": CONFIRMATION_POLICY_LABELS[confirmation_policy_label_value],
        "prompt_injection_guard": prompt_injection_guard_value,
        "streaming_enabled": streaming_enabled_value,
        "plan_progress_enabled": plan_progress_enabled_value,
        "trace_level": trace_level_value,
        "max_tool_calls_per_run": max_tool_calls_per_run_value,
        "max_web_pages_per_run": max_web_pages_per_run_value,
    }


def build_compare_config_snapshot(config):
    return {
        "run_mode": config["run_mode"],
        "multi_agent_architecture_label": config["multi_agent_architecture_label"],
        "multi_agent_architecture": config["multi_agent_architecture"],
        "source_strategy_label": config["source_strategy_label"],
        "router_mode": config["router_mode"],
        "source_strategy": config["source_strategy"],
        "retrieval_strategy_label": config["retrieval_strategy_label"],
        "retrieval_strategy": config["retrieval_strategy"],
        "context_packing_label": config["context_packing_label"],
        "context_packing_strategy": config["context_packing_strategy"],
        "chunking_strategy": config["chunking_strategy"],
        "chunking_strategy_labels": config["chunking_strategy_labels"],
        "deepseek_model": config["deepseek_model"],
        "deepseek_model_label": config["deepseek_model_label"],
        "planner_type": config["planner_type"],
        "evaluator_type": config["evaluator_type"],
        "memory_enabled": config["memory_enabled"],
        "memory_route_strategy": config["memory_route_strategy"],
        "memory_write_mode": config["memory_write_mode"],
        "streaming_enabled": config["streaming_enabled"],
        "plan_progress_enabled": config["plan_progress_enabled"],
        "reranker_enabled": agent.ENABLE_RERANKER,
        "top_k": config["top_k"],
        "web_max_results": config["web_max_results"],
        "max_autonomous_steps": config["max_autonomous_steps"],
        "safety_mode": config["safety_mode"],
        "confirmation_policy": config["confirmation_policy"],
        "prompt_injection_guard": config["prompt_injection_guard"],
        "max_tool_calls_per_run": config["max_tool_calls_per_run"],
        "max_web_pages_per_run": config["max_web_pages_per_run"],
    }


def latest_compare_run(panel_id):
    messages = st.session_state.get(compare_state_key(panel_id, "messages"), [])
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("badcase_run"):
            return message["badcase_run"]
    return None


def render_compare_diff_summary():
    left_run = latest_compare_run("left")
    right_run = latest_compare_run("right")
    if not left_run or not right_run:
        return

    same_question = left_run.get("user_input") == right_run.get("user_input")
    left_tools = left_run.get("tools_called", [])
    right_tools = right_run.get("tools_called", [])
    left_sources = summarize_source_types(left_run.get("sources", []))
    right_sources = summarize_source_types(right_run.get("sources", []))
    left_source_titles = {source.get("source", "") for source in left_run.get("sources", []) if source.get("source")}
    right_source_titles = {source.get("source", "") for source in right_run.get("sources", []) if source.get("source")}
    left_unique_sources = "、".join(sorted(left_source_titles - right_source_titles)[:3]) or "无"
    right_unique_sources = "、".join(sorted(right_source_titles - left_source_titles)[:3]) or "无"
    left_config = left_run.get("config", {})
    right_config = right_run.get("config", {})

    with st.expander("A/B 差异摘要", expanded=True):
        if same_question:
            st.caption(f"对照问题：{left_run.get('user_input', '')}")
        else:
            st.caption("左右两侧最近一轮问题不同，以下只做运行差异对比。")
        rows = [
            ("资料来源", left_sources, right_sources),
            ("独有来源", left_unique_sources, right_unique_sources),
            ("调用工具", "、".join(left_tools) or "无", "、".join(right_tools) or "无"),
            ("资料策略", left_config.get("source_strategy", ""), right_config.get("source_strategy", "")),
            ("检索策略", left_config.get("retrieval_strategy", ""), right_config.get("retrieval_strategy", "")),
            ("上下文策略", left_config.get("context_packing_strategy", ""), right_config.get("context_packing_strategy", "")),
            ("模型", left_config.get("deepseek_model_label", ""), right_config.get("deepseek_model_label", "")),
            ("耗时", f"{left_run.get('elapsed_ms', 0)} ms", f"{right_run.get('elapsed_ms', 0)} ms"),
            ("回答长度", f"{len(str(left_run.get('actual_answer', '')))} 字", f"{len(str(right_run.get('actual_answer', '')))} 字"),
        ]
        st.table([
            {"对比项": label, "Agent A": left_value, "Agent B": right_value}
            for label, left_value, right_value in rows
        ])


def run_compare_agent_turn(panel_id, prompt, config):
    messages_key = compare_state_key(panel_id, "messages")
    session_id = st.session_state[compare_state_key(panel_id, "rag_session_id")]
    trace_id = generate_trace_id()
    started_at = time.perf_counter()
    conversation_context = build_conversation_context(st.session_state[messages_key])
    st.session_state[messages_key].append({"role": "user", "content": prompt})

    plan_placeholder = st.empty()
    live_plan_steps = base_plan_steps(config["run_mode"])
    if config["plan_progress_enabled"]:
        render_plan_progress(plan_placeholder, live_plan_steps)

    def handle_plan_progress(event):
        if not config["plan_progress_enabled"]:
            return
        merge_plan_event(live_plan_steps, event)
        render_plan_progress(plan_placeholder, live_plan_steps)

    stream_placeholder = st.empty()
    streamed_answer = {"text": ""}

    def handle_answer_stream(delta, full_text):
        streamed_answer["text"] = full_text
        stream_placeholder.markdown(full_text + "▌")

    uploaded_sources = ingest_uploaded_files_for_state(
        config["uploaded_files"],
        config["chunking_strategy"],
        state_key_prefix=f"compare_{panel_id}_",
        session_id=session_id,
    )
    memory_context = ""
    retrieved_memories = []
    memory_route = {}

    use_autonomous_mode = False
    autonomous_route_reason = ""
    if config["run_mode"] == "自主任务":
        use_autonomous_mode, autonomous_route_reason = call_with_supported_kwargs(
            autonomous_agent.should_use_autonomous_mode,
            prompt,
            router_mode=config["router_mode"],
        )

    if config["run_mode"] == "自主任务" and use_autonomous_mode:
        memory_context, retrieved_memories, memory_route = load_routed_memory(
            prompt,
            enabled=config["memory_enabled"],
            conversation_context=conversation_context,
            route_strategy=config["memory_route_strategy"],
        )
        result = call_with_supported_kwargs(
            autonomous_agent.run_autonomous_agent,
            prompt,
            top_k=config["top_k"],
            web_max_results=config["web_max_results"],
            max_steps=config["max_autonomous_steps"],
            preferred_sources=uploaded_sources,
            router_mode=config["router_mode"],
            source_strategy=config["source_strategy"],
            retrieval_strategy=config["retrieval_strategy"],
            context_packing_strategy=config["context_packing_strategy"],
            planner_type=config["planner_type"],
            evaluator_type=config["evaluator_type"],
            memory_context=memory_context,
            memory_enabled=config["memory_enabled"],
            memory_route_strategy=config["memory_route_strategy"],
            multi_agent_architecture=config["multi_agent_architecture"],
            debate_rounds=config.get("debate_rounds", 2),
            conversation_context=conversation_context,
            metadata_scope={"session_id": session_id},
            progress_callback=handle_plan_progress,
            permission_context=permission_context_from_config(config, trace_id),
            trace_id=trace_id,
            model_name=config["deepseek_model"],
        )
    else:
        result = call_with_supported_kwargs(
            agent_runtime.run_agent_pro,
            prompt,
            use_web=True,
            top_k=config["top_k"],
            web_max_results=config["web_max_results"],
            preferred_sources=uploaded_sources,
            router_mode=config["router_mode"],
            source_strategy=config["source_strategy"],
            retrieval_strategy=config["retrieval_strategy"],
            context_packing_strategy=config["context_packing_strategy"],
            planner_type=config["planner_type"],
            evaluator_type=config["evaluator_type"],
            memory_context=memory_context,
            memory_enabled=config["memory_enabled"],
            memory_route_strategy=config["memory_route_strategy"],
            multi_agent_architecture=config["multi_agent_architecture"],
            debate_rounds=config.get("debate_rounds", 2),
            conversation_context=conversation_context,
            metadata_scope={"session_id": session_id},
            stream_callback=handle_answer_stream if config["streaming_enabled"] else None,
            progress_callback=handle_plan_progress,
            permission_context=permission_context_from_config(config, trace_id),
            trace_id=trace_id,
            model_name=config["deepseek_model"],
        )
        if config["run_mode"] == "自主任务":
            handle_plan_progress({
                "id": "goal_manager",
                "name": "自主模式入口判断",
                "tool": "goal_router",
                "status": "completed",
                "summary": f"已回退普通问答：{autonomous_route_reason}",
            })
            result["planner_mode"] = "autonomous_fallback"
            result["steps"] = [
                {
                    "name": "自主模式入口判断",
                    "tool": "goal_router",
                    "reason": "Goal Manager 先判断输入是否值得进入任务级 Autonomous Runtime。",
                    "status": "success",
                    "summary": f"已回退普通问答：{autonomous_route_reason}",
                    "elapsed_ms": 0,
                    "error": "",
                },
                *result.get("steps", []),
            ]

    if streamed_answer["text"]:
        stream_placeholder.empty()

    planner_label = (
        "Autonomous Runtime（自主任务运行时）"
        if result.get("planner_mode") == "autonomous_runtime"
        else "LLM Tool Calling（大模型工具调用）"
        if result.get("planner_mode") == "llm_tool_calling"
        else "自主模式回退普通问答"
        if result.get("planner_mode") == "autonomous_fallback"
        else "行业主流 Runtime（运行时）雏形"
        if result.get("planner_mode") == "pro_runtime"
        else "Multi-Agent（多智能体）教学架构"
        if str(result.get("planner_mode", "")).startswith("multi_agent_")
        else "规则兜底"
    )
    if result.get("multi_agent_architecture"):
        planner_label = f"{planner_label}｜Multi-Agent：{result.get('multi_agent_architecture')}"
    autonomous_snapshot = {}
    if result.get("planner_mode") == "autonomous_runtime":
        goal = result.get("goal")
        autonomous_snapshot = {
            "goal": getattr(goal, "objective", "") if goal else "",
            "stop_reason": result.get("stop_reason", ""),
            "tasks": [
                {
                    "id": getattr(task, "id", ""),
                    "title": getattr(task, "title", ""),
                    "status": getattr(task, "status", ""),
                    "depends_on": getattr(task, "depends_on", []),
                    "expected_output": getattr(task, "expected_output", ""),
                }
                for task in result.get("tasks", [])
            ],
            "critic_results": result.get("critic_results", []),
            "reflections": result.get("reflections", []),
        }
    badcase_run = {
        "trace_id": trace_id,
        "user_input": prompt,
        "actual_answer": result["answer"],
        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "config": build_compare_config_snapshot(config),
        "tools_called": extract_tools_from_steps(result.get("steps", [])),
        "sources_used": extract_source_types(result.get("sources", [])),
        "planner_mode": result.get("planner_mode", ""),
        "planner_label": planner_label,
        "trace_level": config["trace_level"],
        "steps": result.get("steps", []),
        "sources": result.get("sources", []),
        "permission_trace": result.get("permission_trace", []),
        "autonomous": autonomous_snapshot,
        "memory_used": result.get("memory_used", [item.get("id") for item in retrieved_memories]),
        "memory_route": result.get("memory_route", memory_route),
        "run_snapshot": {
            "trace_id": trace_id,
            "planner_mode": result.get("planner_mode", ""),
            "tools_called": extract_tools_from_steps(result.get("steps", [])),
            "sources_used": extract_source_types(result.get("sources", [])),
            "memory_used": result.get("memory_used", [item.get("id") for item in retrieved_memories]),
            "memory_route": result.get("memory_route", memory_route),
            "steps": compact_steps_for_log(result.get("steps", [])),
            "sources": compact_sources_for_log(result.get("sources", [])),
            "answer_preview": str(result.get("answer", ""))[:1200],
        },
    }
    persist_trace_for_run(badcase_run, panel_id=panel_id)
    st.session_state[compare_state_key(panel_id, "last_sources")] = result["sources"]
    st.session_state[compare_state_key(panel_id, "last_agent_run")] = badcase_run
    st.session_state[messages_key].append({
        "role": "assistant",
        "content": result["answer"],
        "badcase_run": badcase_run,
    })
    if config["memory_enabled"] and config["memory_write_mode"] == "confirm":
        candidates = memory_manager.suggest_memory_candidates(prompt)
        for candidate in candidates:
            candidate["candidate_id"] = memory_manager.candidate_id(candidate)
        st.session_state[compare_state_key(panel_id, "pending_memory_candidates")] = candidates


def execute_compare_agent_backend(
    panel_id,
    prompt,
    config,
    *,
    session_id,
    uploaded_sources,
    conversation_context,
):
    trace_id = generate_trace_id()
    started_at = time.perf_counter()
    memory_context = ""
    retrieved_memories = []
    memory_route = {}

    use_autonomous_mode = False
    autonomous_route_reason = ""
    if config["run_mode"] == "自主任务":
        use_autonomous_mode, autonomous_route_reason = call_with_supported_kwargs(
            autonomous_agent.should_use_autonomous_mode,
            prompt,
            router_mode=config["router_mode"],
        )

    if config["run_mode"] == "自主任务" and use_autonomous_mode:
        memory_context, retrieved_memories, memory_route = load_routed_memory(
            prompt,
            enabled=config["memory_enabled"],
            conversation_context=conversation_context,
            route_strategy=config["memory_route_strategy"],
        )
        result = call_with_supported_kwargs(
            autonomous_agent.run_autonomous_agent,
            prompt,
            top_k=config["top_k"],
            web_max_results=config["web_max_results"],
            max_steps=config["max_autonomous_steps"],
            preferred_sources=uploaded_sources,
            router_mode=config["router_mode"],
            source_strategy=config["source_strategy"],
            retrieval_strategy=config["retrieval_strategy"],
            context_packing_strategy=config["context_packing_strategy"],
            planner_type=config["planner_type"],
            evaluator_type=config["evaluator_type"],
            memory_context=memory_context,
            memory_enabled=config["memory_enabled"],
            memory_route_strategy=config["memory_route_strategy"],
            multi_agent_architecture=config["multi_agent_architecture"],
            debate_rounds=config.get("debate_rounds", 2),
            conversation_context=conversation_context,
            metadata_scope={"session_id": session_id},
            permission_context=permission_context_from_config(config, trace_id),
            trace_id=trace_id,
            model_name=config["deepseek_model"],
        )
    else:
        result = call_with_supported_kwargs(
            agent_runtime.run_agent_pro,
            prompt,
            use_web=True,
            top_k=config["top_k"],
            web_max_results=config["web_max_results"],
            preferred_sources=uploaded_sources,
            router_mode=config["router_mode"],
            source_strategy=config["source_strategy"],
            retrieval_strategy=config["retrieval_strategy"],
            context_packing_strategy=config["context_packing_strategy"],
            planner_type=config["planner_type"],
            evaluator_type=config["evaluator_type"],
            memory_context=memory_context,
            memory_enabled=config["memory_enabled"],
            memory_route_strategy=config["memory_route_strategy"],
            multi_agent_architecture=config["multi_agent_architecture"],
            debate_rounds=config.get("debate_rounds", 2),
            conversation_context=conversation_context,
            metadata_scope={"session_id": session_id},
            permission_context=permission_context_from_config(config, trace_id),
            trace_id=trace_id,
            model_name=config["deepseek_model"],
        )
        if config["run_mode"] == "自主任务":
            result["planner_mode"] = "autonomous_fallback"
            result["steps"] = [
                {
                    "name": "自主模式入口判断",
                    "tool": "goal_router",
                    "reason": "Goal Manager 先判断输入是否值得进入任务级 Autonomous Runtime。",
                    "status": "success",
                    "summary": f"已回退普通问答：{autonomous_route_reason}",
                    "elapsed_ms": 0,
                    "error": "",
                },
                *result.get("steps", []),
            ]

    planner_label = (
        "Autonomous Runtime（自主任务运行时）"
        if result.get("planner_mode") == "autonomous_runtime"
        else "LLM Tool Calling（大模型工具调用）"
        if result.get("planner_mode") == "llm_tool_calling"
        else "自主模式回退普通问答"
        if result.get("planner_mode") == "autonomous_fallback"
        else "行业主流 Runtime（运行时）雏形"
        if result.get("planner_mode") == "pro_runtime"
        else "Multi-Agent（多智能体）教学架构"
        if str(result.get("planner_mode", "")).startswith("multi_agent_")
        else "规则兜底"
    )
    if result.get("multi_agent_architecture"):
        planner_label = f"{planner_label}｜Multi-Agent：{result.get('multi_agent_architecture')}"
    autonomous_snapshot = {}
    if result.get("planner_mode") == "autonomous_runtime":
        goal = result.get("goal")
        autonomous_snapshot = {
            "goal": getattr(goal, "objective", "") if goal else "",
            "stop_reason": result.get("stop_reason", ""),
            "tasks": [
                {
                    "id": getattr(task, "id", ""),
                    "title": getattr(task, "title", ""),
                    "status": getattr(task, "status", ""),
                    "depends_on": getattr(task, "depends_on", []),
                    "expected_output": getattr(task, "expected_output", ""),
                }
                for task in result.get("tasks", [])
            ],
            "critic_results": result.get("critic_results", []),
            "reflections": result.get("reflections", []),
        }

    badcase_run = {
        "trace_id": trace_id,
        "user_input": prompt,
        "actual_answer": result["answer"],
        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
        "config": build_compare_config_snapshot(config),
        "tools_called": extract_tools_from_steps(result.get("steps", [])),
        "sources_used": extract_source_types(result.get("sources", [])),
        "planner_mode": result.get("planner_mode", ""),
        "planner_label": planner_label,
        "trace_level": config["trace_level"],
        "steps": result.get("steps", []),
        "sources": result.get("sources", []),
        "permission_trace": result.get("permission_trace", []),
        "autonomous": autonomous_snapshot,
        "memory_used": result.get("memory_used", [item.get("id") for item in retrieved_memories]),
        "memory_route": result.get("memory_route", memory_route),
        "run_snapshot": {
            "trace_id": trace_id,
            "planner_mode": result.get("planner_mode", ""),
            "tools_called": extract_tools_from_steps(result.get("steps", [])),
            "sources_used": extract_source_types(result.get("sources", [])),
            "memory_used": result.get("memory_used", [item.get("id") for item in retrieved_memories]),
            "memory_route": result.get("memory_route", memory_route),
            "steps": compact_steps_for_log(result.get("steps", [])),
            "sources": compact_sources_for_log(result.get("sources", [])),
            "answer_preview": str(result.get("answer", ""))[:1200],
        },
    }

    pending_memory_candidates = []
    if config["memory_enabled"] and config["memory_write_mode"] == "confirm":
        pending_memory_candidates = memory_manager.suggest_memory_candidates(prompt)
        for candidate in pending_memory_candidates:
            candidate["candidate_id"] = memory_manager.candidate_id(candidate)

    return {
        "panel_id": panel_id,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "badcase_run": badcase_run,
        "pending_memory_candidates": pending_memory_candidates,
    }


def apply_compare_agent_backend_result(panel_id, backend_result):
    messages_key = compare_state_key(panel_id, "messages")
    badcase_run = backend_result["badcase_run"]
    persist_trace_for_run(badcase_run, panel_id=panel_id)
    st.session_state[compare_state_key(panel_id, "last_sources")] = backend_result["sources"]
    st.session_state[compare_state_key(panel_id, "last_agent_run")] = badcase_run
    st.session_state[messages_key].append({
        "role": "assistant",
        "content": backend_result["answer"],
        "badcase_run": badcase_run,
    })
    if backend_result["pending_memory_candidates"]:
        st.session_state[compare_state_key(panel_id, "pending_memory_candidates")] = backend_result["pending_memory_candidates"]


def render_compare_agent_workspace(panel_id, title, config):
    st.markdown(
        '<div class="config-summary"><div class="config-summary-title">当前配置</div>'
        f'<span class="config-pill">{config["run_mode"]}</span>'
        f'<span class="config-pill">{config["source_strategy_label"]}</span>'
        f'<span class="config-pill">{config["retrieval_strategy_label"]}</span>'
        f'<span class="config-pill">{config["context_packing_label"]}</span>'
        f'<span class="config-pill">{config["deepseek_model_label"]}</span>'
        f'<span class="config-pill">{config["safety_mode_label"]}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_left, action_right = st.columns([0.62, 0.38])
    with action_right:
        if st.button(f"清空 {title} 对话", key=f"clear_compare_{panel_id}", use_container_width=True):
            st.session_state[compare_state_key(panel_id, "messages")] = []
            st.session_state[compare_state_key(panel_id, "last_sources")] = []
            st.session_state[compare_state_key(panel_id, "last_agent_run")] = None
            st.session_state[compare_state_key(panel_id, "pending_memory_candidates")] = []
            st.rerun()

    with st.form(f"compare_prompt_form_{panel_id}", clear_on_submit=True):
        prompt = st.text_area(
            f"{title} 输入问题",
            placeholder=f"输入问题，{title} 会按本侧配置检索上传资料和网络资料",
            key=f"compare_prompt_{panel_id}",
            height=96,
            disabled=not deepseek_key,
        )
        submitted = st.form_submit_button(
            f"发送给 {title}",
            use_container_width=True,
            disabled=not deepseek_key,
        )

    if submitted:
        prompt = prompt.strip()
        if not prompt:
            st.warning("请输入问题。")
        elif not deepseek_key:
            st.error("请先配置 DEEPSEEK_API_KEY。")
        else:
            with st.spinner(f"{title} 执行中..."):
                try:
                    run_compare_agent_turn(panel_id, prompt, config)
                except Exception as error:
                    trace_id = generate_trace_id()
                    error_answer = f"调用失败：{error}"
                    error_run = {
                        "trace_id": trace_id,
                        "user_input": prompt,
                        "actual_answer": error_answer,
                        "config": build_compare_config_snapshot(config),
                        "tools_called": [],
                        "sources_used": [],
                        "planner_mode": "error",
                        "planner_label": "错误",
                        "trace_level": config["trace_level"],
                        "steps": [],
                        "sources": [],
                        "permission_trace": [],
                        "memory_used": [],
                        "run_snapshot": {
                            "trace_id": trace_id,
                            "planner_mode": "error",
                            "error": str(error)[:1200],
                            "answer_preview": error_answer,
                        },
                    }
                    persist_trace_for_run(error_run, panel_id=panel_id, status="error", error=str(error))
                    st.session_state[compare_state_key(panel_id, "messages")].append({
                        "role": "assistant",
                        "content": error_answer,
                        "badcase_run": error_run,
                    })

    messages = st.session_state[compare_state_key(panel_id, "messages")]
    if not messages:
        st.markdown(
            '<div class="empty-state"><strong>可以直接开始对照测试</strong>'
            '<div class="compact-help">左右两侧配置互相独立；适合用同一个问题比较不同策略。</div></div>',
            unsafe_allow_html=True,
        )
        return
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(
                    message["content"],
                    message.get("badcase_run"),
                    key_suffix=f"compare_{panel_id}_{index}",
                )
            else:
                st.write(message["content"])

    pending_candidates = st.session_state.get(compare_state_key(panel_id, "pending_memory_candidates"), [])
    if pending_candidates:
        with st.expander("Memory（记忆）候选", expanded=False):
            st.caption("双 Agent 对照模式暂只展示候选，不在这里写入长期记忆。需要写入时请回到单 Agent 模式确认。")
            for candidate in pending_candidates:
                st.write(candidate.get("value", ""))
                st.caption(f"key：{candidate.get('key', '')}｜confidence：{candidate.get('confidence', 0):.2f}")


def render_compare_agent_panel(panel_id, title):
    ensure_compare_state(panel_id)
    config = render_compare_settings(panel_id, title)
    render_compare_agent_workspace(panel_id, title, config)


def render_dual_agent_compare_mode():
    ensure_compare_state("left")
    ensure_compare_state("right")
    st.markdown(
        '<div class="main-header-card"><h1>双 Agent 对照模式</h1>'
        '<p>左右两侧是两个独立 Agent 实例。分别配置、上传、提问，用来观察不同策略带来的回答差异。</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="workspace-note"><p>状态隔离：两侧各自拥有独立 session_id、上传入库记录、对话历史、执行 trace 和 badcase 现场。</p></div>',
        unsafe_allow_html=True,
    )
    if not deepseek_key:
        st.warning("DeepSeek API Key 未配置，双 Agent 暂不能发送问题。请先在 Streamlit Secrets 中配置 DEEPSEEK_API_KEY。")
    else:
        st.caption("运行检查：DeepSeek 已配置，可以开始对照测试。")
    settings_left, settings_right = st.columns(2, gap="large")
    with settings_left:
        with st.expander("Agent A 配置", expanded=False):
            left_config = render_compare_settings("left", "Agent A")
    with settings_right:
        with st.expander("Agent B 配置", expanded=False):
            right_config = render_compare_settings("right", "Agent B")

    with st.form("compare_shared_prompt_form", clear_on_submit=True):
        shared_prompt = st.text_area(
            "同一问题并行发送给 A/B",
            placeholder="输入一个问题，让 Agent A 和 Agent B 按各自配置同时回答，便于直接对比差异。",
            key="compare_shared_prompt",
            height=88,
            disabled=not deepseek_key,
        )
        shared_submitted = st.form_submit_button(
            "并行发送给 Agent A 和 Agent B",
            use_container_width=True,
            disabled=not deepseek_key,
        )

    if shared_submitted:
        shared_prompt = shared_prompt.strip()
        if not shared_prompt:
            st.warning("请输入要同时发送的问题。")
        elif not deepseek_key:
            st.error("请先配置 DEEPSEEK_API_KEY。")
        else:
            jobs = {}
            for panel_id, config in [("left", left_config), ("right", right_config)]:
                messages_key = compare_state_key(panel_id, "messages")
                session_id = st.session_state[compare_state_key(panel_id, "rag_session_id")]
                conversation_context = build_conversation_context(st.session_state[messages_key])
                st.session_state[messages_key].append({"role": "user", "content": shared_prompt})
                uploaded_sources = ingest_uploaded_files_for_state(
                    config["uploaded_files"],
                    config["chunking_strategy"],
                    state_key_prefix=f"compare_{panel_id}_",
                    session_id=session_id,
                )
                jobs[panel_id] = {
                    "config": config,
                    "session_id": session_id,
                    "conversation_context": conversation_context,
                    "uploaded_sources": uploaded_sources,
                }

            with st.spinner("Agent A / B 正在并行执行..."):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_panel = {
                        executor.submit(
                            execute_compare_agent_backend,
                            panel_id,
                            shared_prompt,
                            job["config"],
                            session_id=job["session_id"],
                            uploaded_sources=job["uploaded_sources"],
                            conversation_context=job["conversation_context"],
                        ): panel_id
                        for panel_id, job in jobs.items()
                    }
                    for future in as_completed(future_to_panel):
                        panel_id = future_to_panel[future]
                        title = "Agent A" if panel_id == "left" else "Agent B"
                        try:
                            apply_compare_agent_backend_result(panel_id, future.result())
                        except Exception as error:
                            st.error(f"{title} 调用失败：{error}")

    left, right = st.columns(2, gap="large")
    with left:
        render_compare_agent_workspace("left", "Agent A", left_config)
    with right:
        render_compare_agent_workspace("right", "Agent B", right_config)
    render_compare_diff_summary()
    render_badcase_form()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_session_id" not in st.session_state:
    st.session_state.rag_session_id = f"session_{uuid4().hex[:12]}"

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "ingested_uploads" not in st.session_state:
    st.session_state.ingested_uploads = {}

if "upload_status" not in st.session_state:
    st.session_state.upload_status = []

if "last_agent_run" not in st.session_state:
    st.session_state.last_agent_run = None

if "show_badcase_form" not in st.session_state:
    st.session_state.show_badcase_form = False

if "pending_memory_candidates" not in st.session_state:
    st.session_state.pending_memory_candidates = []

if "dismissed_memory_candidates" not in st.session_state:
    st.session_state.dismissed_memory_candidates = set()

if "memory_notice" not in st.session_state:
    st.session_state.memory_notice = ""

if "prompt_seed" not in st.session_state:
    st.session_state.prompt_seed = ""

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = ""


st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1180px;}
[data-testid="stSidebar"] {
    border-right: 1px solid #e2e6ee;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}
.main-header-card {
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    background: #ffffff;
    box-shadow: 0 10px 28px rgba(23, 31, 56, 0.06);
    margin-bottom: 0.9rem;
}
.main-header-card h1 {font-size: 1.7rem; margin: 0 0 0.25rem 0;}
.main-header-card p {margin: 0; color: #6b7280;}
.settings-panel, .observe-panel {
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    background: #ffffff;
    padding: 0.9rem;
    box-shadow: 0 10px 28px rgba(23, 31, 56, 0.06);
}
.settings-panel {margin-bottom: 0.9rem;}
.workspace-note {
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    background: #ffffff;
    padding: 0.8rem 1rem;
    margin-bottom: 0.9rem;
}
.workspace-note p {margin: 0; color: #6b7280;}
.badcase-inline button {
    border-radius: 7px !important;
}
.workspace-composer {
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    background: #ffffff;
    padding: 0.8rem;
    margin-top: 1rem;
    box-shadow: 0 10px 28px rgba(23, 31, 56, 0.06);
}
.config-summary {
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    background: #ffffff;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 10px 28px rgba(23, 31, 56, 0.06);
}
.run-config-snapshot {
    border: 1px solid #e2e6ee;
    border-radius: 8px;
    background: #fbfcfe;
    padding: 0.55rem 0.65rem;
    margin: 0.55rem 0 0.25rem 0;
}
.snapshot-title {
    display: inline-block;
    font-weight: 700;
    font-size: 0.82rem;
    margin-right: 0.35rem;
    color: #374151;
}
.config-summary-title {
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.config-pill {
    display: inline-block;
    border: 1px solid #d7dfeb;
    border-radius: 999px;
    padding: 0.22rem 0.55rem;
    margin: 0.18rem 0.2rem 0.18rem 0;
    background: #f8fafc;
    color: #4b5563;
    font-size: 0.82rem;
}
.empty-state {
    border: 1px dashed #cfd6e3;
    border-radius: 8px;
    background: #fbfcfe;
    padding: 1rem;
    margin-bottom: 0.9rem;
}
.empty-state strong {
    display: block;
    margin-bottom: 0.35rem;
}
.compact-help {
    color: #6b7280;
    font-size: 0.9rem;
}
@media (max-width: 768px) {
    .block-container {padding-left: 1rem; padding-right: 1rem;}
    .main-header-card h1 {font-size: 1.35rem;}
    .config-pill {font-size: 0.78rem;}
    .run-config-snapshot {padding: 0.5rem;}
}
</style>
""", unsafe_allow_html=True)

page_mode = st.radio(
    "页面模式",
    ["单 Agent", "双 Agent 对照"],
    horizontal=True,
    help="双 Agent 对照模式会把当前 Agent 工作台复制成左右两个独立实例。",
)

if page_mode == "双 Agent 对照":
    render_dual_agent_compare_mode()
    st.stop()

with st.sidebar:
    render_settings_panel()

st.markdown(
    '<div class="main-header-card"><h1>agent for train</h1>'
    '<p>提问、观察执行过程、检查证据来源、反馈 badcase。资料与教学配置在左侧栏。</p></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="config-summary"><div class="config-summary-title">本轮配置</div>'
    f'<span class="config-pill">{run_mode}</span>'
    f'<span class="config-pill">Multi-Agent：{multi_agent_architecture_label}</span>'
    f'<span class="config-pill">{source_strategy_label}</span>'
    f'<span class="config-pill">{retrieval_strategy_label}</span>'
    f'<span class="config-pill">{context_packing_label}</span>'
    f'<span class="config-pill">{deepseek_model_label}</span>'
    f'<span class="config-pill">{safety_mode_label}</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state"><strong>可以直接开始提问</strong>'
        '<div class="compact-help">上传资料后会优先检索上传内容；没有上传资料时，会按当前资料来源策略联网或使用本地基础资料。</div></div>',
        unsafe_allow_html=True,
    )
    prompt_cols = st.columns(3)
    examples = [
        "你能做些什么？",
        "RAG 是什么？用产品经理能听懂的话解释",
        "最近 AI Agent 有什么新趋势？",
    ]
    for col, example in zip(prompt_cols, examples):
        with col:
            st.button(
                example,
                key=f"prompt_example_{example}",
                on_click=set_prompt_seed,
                args=(example,),
                use_container_width=True,
            )

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(
                message["content"],
                message.get("badcase_run"),
                key_suffix=f"history_{index}",
            )
        else:
            st.write(message["content"])


prompt = st.session_state.queued_prompt.strip()
st.session_state.queued_prompt = ""
chat_prompt = st.chat_input("输入问题，Agent（智能体）会自动检索上传资料和网络资料")
if chat_prompt:
    prompt = chat_prompt.strip()

if prompt:
    if not deepseek_key:
        st.error("请先配置 DEEPSEEK_API_KEY。")
        st.stop()

    trace_id = generate_trace_id()
    started_at = time.perf_counter()
    conversation_context = build_conversation_context(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            plan_placeholder = st.empty()
            live_plan_steps = base_plan_steps(run_mode)
            if plan_progress_enabled:
                render_plan_progress(plan_placeholder, live_plan_steps)

            def handle_plan_progress(event):
                if not plan_progress_enabled:
                    return
                merge_plan_event(live_plan_steps, event)
                render_plan_progress(plan_placeholder, live_plan_steps)

            stream_placeholder = st.empty()
            streamed_answer = {"text": ""}

            def handle_answer_stream(delta, full_text):
                streamed_answer["text"] = full_text
                stream_placeholder.markdown(full_text + "▌")

            with st.spinner("执行 Agent（智能体）计划中..."):
                uploaded_sources = ingest_uploaded_files(uploaded_files, prompt, chunking_strategy)
                memory_context = ""
                retrieved_memories = []
                memory_route = {}

                use_autonomous_mode = False
                autonomous_route_reason = ""
                if run_mode == "自主任务":
                    use_autonomous_mode, autonomous_route_reason = call_with_supported_kwargs(
                        autonomous_agent.should_use_autonomous_mode,
                        prompt,
                        router_mode=router_mode,
                    )

                if run_mode == "自主任务" and use_autonomous_mode:
                    memory_context, retrieved_memories, memory_route = load_routed_memory(
                        prompt,
                        enabled=memory_enabled,
                        conversation_context=conversation_context,
                        route_strategy=memory_route_strategy,
                    )
                    result = call_with_supported_kwargs(
                        autonomous_agent.run_autonomous_agent,
                        prompt,
                        top_k=top_k,
                        web_max_results=web_max_results,
                        max_steps=max_autonomous_steps,
                        preferred_sources=uploaded_sources,
                        router_mode=router_mode,
                        source_strategy=source_strategy,
                        retrieval_strategy=retrieval_strategy,
                        context_packing_strategy=context_packing_strategy,
                        planner_type=planner_type,
                        evaluator_type=evaluator_type,
                        memory_context=memory_context,
                        memory_enabled=memory_enabled,
                        memory_route_strategy=memory_route_strategy,
                        multi_agent_architecture=multi_agent_architecture,
                        debate_rounds=debate_rounds,
                        conversation_context=conversation_context,
                        metadata_scope={"session_id": st.session_state.rag_session_id},
                        progress_callback=handle_plan_progress,
                        permission_context=current_permission_context({"trace_id": trace_id}),
                        trace_id=trace_id,
                        model_name=deepseek_model,
                    )
                else:
                    answer_stream_callback = handle_answer_stream if streaming_enabled else None
                    result = call_with_supported_kwargs(
                        agent_runtime.run_agent_pro,
                        prompt,
                        use_web=True,
                        top_k=top_k,
                        web_max_results=web_max_results,
                        preferred_sources=uploaded_sources,
                        router_mode=router_mode,
                        source_strategy=source_strategy,
                        retrieval_strategy=retrieval_strategy,
                        context_packing_strategy=context_packing_strategy,
                        planner_type=planner_type,
                        evaluator_type=evaluator_type,
                        memory_context=memory_context,
                        memory_enabled=memory_enabled,
                        memory_route_strategy=memory_route_strategy,
                        multi_agent_architecture=multi_agent_architecture,
                        debate_rounds=debate_rounds,
                        conversation_context=conversation_context,
                        metadata_scope={"session_id": st.session_state.rag_session_id},
                        stream_callback=answer_stream_callback,
                        progress_callback=handle_plan_progress,
                        permission_context=current_permission_context({"trace_id": trace_id}),
                        trace_id=trace_id,
                        model_name=deepseek_model,
                    )
                    if run_mode == "自主任务":
                        handle_plan_progress({
                            "id": "goal_manager",
                            "name": "自主模式入口判断",
                            "tool": "goal_router",
                            "status": "completed",
                            "summary": f"已回退普通问答：{autonomous_route_reason}",
                        })
                        result["planner_mode"] = "autonomous_fallback"
                        result["steps"] = [
                            {
                                "name": "自主模式入口判断",
                                "tool": "goal_router",
                                "reason": "Goal Manager（目标管理器）先判断输入是否值得进入任务级 Autonomous Runtime（自主任务运行时）。",
                                "status": "success",
                                "summary": f"已回退普通问答：{autonomous_route_reason}",
                                "elapsed_ms": 0,
                                "error": "",
                            },
                            *result.get("steps", []),
                        ]

            if streamed_answer["text"]:
                stream_placeholder.empty()
            planner_label = (
                "Autonomous Runtime（自主任务运行时）"
                if result.get("planner_mode") == "autonomous_runtime"
                else "LLM Tool Calling（大模型工具调用）"
                if result.get("planner_mode") == "llm_tool_calling"
                else "自主模式回退普通问答"
                if result.get("planner_mode") == "autonomous_fallback"
                else "行业主流 Runtime（运行时）雏形"
                if result.get("planner_mode") == "pro_runtime"
                else "Multi-Agent（多智能体）教学架构"
                if str(result.get("planner_mode", "")).startswith("multi_agent_")
                else "规则兜底"
            )
            if result.get("multi_agent_architecture"):
                planner_label = f"{planner_label}｜Multi-Agent：{result.get('multi_agent_architecture')}"
            autonomous_snapshot = {}
            if result.get("planner_mode") == "autonomous_runtime":
                goal = result.get("goal")
                autonomous_snapshot = {
                    "goal": getattr(goal, "objective", "") if goal else "",
                    "stop_reason": result.get("stop_reason", ""),
                    "tasks": [
                        {
                            "id": getattr(task, "id", ""),
                            "title": getattr(task, "title", ""),
                            "status": getattr(task, "status", ""),
                            "depends_on": getattr(task, "depends_on", []),
                            "expected_output": getattr(task, "expected_output", ""),
                        }
                        for task in result.get("tasks", [])
                    ],
                    "critic_results": result.get("critic_results", []),
                    "reflections": result.get("reflections", []),
                }
            badcase_run = {
                "trace_id": trace_id,
                "user_input": prompt,
                "actual_answer": result["answer"],
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                "config": build_current_config(),
                "tools_called": extract_tools_from_steps(result.get("steps", [])),
                "sources_used": extract_source_types(result.get("sources", [])),
                "planner_mode": result.get("planner_mode", ""),
                "planner_label": planner_label,
                "trace_level": trace_level,
                "steps": result.get("steps", []),
                "sources": result.get("sources", []),
                "permission_trace": result.get("permission_trace", []),
                "autonomous": autonomous_snapshot,
                "memory_used": result.get("memory_used", [item.get("id") for item in retrieved_memories]),
                "memory_route": result.get("memory_route", memory_route),
                "run_snapshot": {
                    "trace_id": trace_id,
                    "planner_mode": result.get("planner_mode", ""),
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    "tools_called": extract_tools_from_steps(result.get("steps", [])),
                    "sources_used": extract_source_types(result.get("sources", [])),
                    "memory_used": result.get("memory_used", [item.get("id") for item in retrieved_memories]),
                    "memory_route": result.get("memory_route", memory_route),
                    "steps": compact_steps_for_log(result.get("steps", [])),
                    "sources": compact_sources_for_log(result.get("sources", [])),
                    "answer_preview": str(result.get("answer", ""))[:1200],
                },
            }
            persist_trace_for_run(badcase_run)
            render_assistant_message(
                result["answer"],
                badcase_run,
                key_suffix=f"live_{len(st.session_state.messages)}",
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "badcase_run": badcase_run,
            })
            st.session_state.last_sources = result["sources"]

            if memory_enabled and memory_write_mode == "confirm":
                candidates = memory_manager.suggest_memory_candidates(prompt)
                for candidate in candidates:
                    candidate["candidate_id"] = memory_manager.candidate_id(candidate)
                st.session_state.pending_memory_candidates = candidates

        except Exception as e:
            error_answer = f"调用失败：{e}"
            error_run = {
                "trace_id": trace_id,
                "user_input": prompt,
                "actual_answer": error_answer,
                "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                "config": build_current_config(),
                "tools_called": [],
                "sources_used": [],
                "planner_mode": "error",
                "planner_label": "错误",
                "trace_level": trace_level,
                "steps": [],
                "sources": [],
                "permission_trace": [],
                "memory_used": [],
                "run_snapshot": {
                    "trace_id": trace_id,
                    "planner_mode": "error",
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    "error": str(e)[:1200],
                    "answer_preview": error_answer,
                },
            }
            persist_trace_for_run(error_run, status="error", error=str(e))
            st.error(error_answer)
            render_assistant_message(error_answer, error_run, key_suffix=f"error_{len(st.session_state.messages)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_answer,
                "badcase_run": error_run,
            })

render_badcase_form()
if st.session_state.memory_notice:
    st.info(st.session_state.memory_notice)
render_memory_confirmation()
