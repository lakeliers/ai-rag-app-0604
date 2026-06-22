import os
import inspect
import hashlib
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

agent.seed_local_note()
if seed_teaching_memory != "0":
    memory_manager.seed_default_memories_if_empty()


st.set_page_config(
    page_title="RAG Agent Pro",
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


def ingest_uploaded_files(uploaded_files, question, chunking_strategy):
    if not uploaded_files:
        return []

    ingested_sources = []
    metadata_scope = {"session_id": st.session_state.rag_session_id}

    for uploaded_file in uploaded_files:
        key = file_key(uploaded_file, chunking_strategy)
        if key in st.session_state.ingested_uploads:
            ingested_sources.append(st.session_state.ingested_uploads[key])
            continue

        if is_image(uploaded_file):
            if not dashscope_key:
                st.warning(f"{uploaded_file.name} 是图片，需要配置 DASHSCOPE_API_KEY 才能解析。")
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

        st.session_state.ingested_uploads[key] = source
        st.session_state.upload_status.append(f"{source}：{chunk_count} 块｜切分：{format_chunking_labels(chunking_strategy)}")
        ingested_sources.append(source)

    return ingested_sources


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


def build_current_config():
    return {
        "run_mode": run_mode,
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
        "memory_write_mode": memory_write_mode,
        "streaming_enabled": streaming_enabled,
        "plan_progress_enabled": plan_progress_enabled,
        "reranker_enabled": agent.ENABLE_RERANKER,
        "top_k": top_k,
        "web_max_results": web_max_results,
        "max_autonomous_steps": max_autonomous_steps,
    }


def set_badcase_target(run):
    st.session_state.last_agent_run = run
    st.session_state.show_badcase_form = True


def render_assistant_message(content, run=None, key_suffix=""):
    left, right = st.columns([0.94, 0.06], vertical_alignment="top")
    with left:
        st.write(content)
        if run and run.get("trace_id"):
            st.caption(f"Trace ID（运行追踪编号）：{run['trace_id']}")
    if run:
        with right:
            st.button(
                "!",
                key=f"badcase_button_{key_suffix}",
                help="反馈 badcase（不良案例）",
                on_click=set_badcase_target,
                args=(run,),
            )


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
        {"id": "memory_retriever", "name": "读取长期记忆", "tool": "Memory", "status": "pending", "summary": "如启用 Memory，会先读取相关记忆。"},
        {"id": "intent_classifier", "name": "意图分类", "tool": "Intent Classifier", "status": "pending", "summary": "判断问题类型。"},
        {"id": "planner", "name": "高层规划", "tool": "Planner", "status": "pending", "summary": "决定走普通回答、上传资料、联网或混合 RAG。"},
        {"id": "orchestrator", "name": "任务编排", "tool": "Orchestrator", "status": "pending", "summary": "把计划展开成可执行节点。"},
        {"id": "web_collect", "name": "联网收集资料", "tool": "Web Collect", "status": "pending", "summary": "需要联网时读取网页正文。"},
        {"id": "rag_search", "name": "RAG 检索", "tool": "RAG Search", "status": "pending", "summary": "检索上传资料、网页资料和本地知识。"},
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
    current_step = next((step for step in plan_steps if step.get("status") == "running"), None)
    if current_step is None:
        current_step = next((step for step in plan_steps if step.get("status") == "failed"), None)
    if current_step is None:
        current_step = next((step for step in plan_steps if step.get("status") == "pending"), None)
    if current_step is None and plan_steps:
        current_step = plan_steps[-1]

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
        if current_step:
            current_status = current_step.get("status", "pending")
            current_label = PLAN_STATUS_LABELS.get(current_status, current_status)
            current_icon = PLAN_STATUS_ICONS.get(current_status, "○")
            current_summary = current_step.get("summary", "")
            st.markdown(
                "**Agent 当前环节**  \n"
                f"{current_icon} **{current_label}**｜{current_step['name']}｜{current_summary}"
            )
        else:
            st.markdown("**Agent 当前环节**  \n○ 等待开始")

        with st.expander(f"查看完整 Plan（计划）执行进度：{completed}/{total} 已完成", expanded=False):
            st.markdown("\n\n".join(rows) if rows else "暂无执行步骤。")


def render_badcase_form():
    run = st.session_state.get("last_agent_run")
    if not run or not st.session_state.get("show_badcase_form"):
        return

    with st.expander("反馈 badcase（不良案例）：补充 Regression Case（回归用例）信息", expanded=True):
        st.markdown("**当前问题现场**")
        if run.get("trace_id"):
            st.code(run["trace_id"], language="text")
        st.write("User Prompt（用户问题）：", run["user_input"])
        st.write("Agent Answer（智能体回答）：", run["actual_answer"])
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
            save_target = st.radio(
                "保存位置",
                badcase_manager.SAVE_TARGETS,
                index=0,
                help="eval 是评估集合；GitHub Issue 是 GitHub 上的问题单，用于开发者确认 badcase（不良案例）。",
            )

            st.markdown("**Case（用例）基础信息**")
            category = st.selectbox(
                "category（问题类型）",
                badcase_manager.CATEGORIES,
                index=badcase_manager.CATEGORIES.index(default_category),
                format_func=lambda value: CATEGORY_LABELS.get(value, value),
                help="category 表示这个 badcase 属于哪类能力问题。",
            )
            case_id = st.text_input(
                "case_id（用例编号）",
                value=badcase_manager.generate_case_id(run["user_input"], category),
                help="case_id 是 regression set 里的唯一用例编号。",
            )
            suite = st.multiselect(
                "suite（评估集合）",
                badcase_manager.SUITES,
                default=["regression"],
                help="suite 表示这个 case 加入哪个测试集合：smoke 冒烟、regression 回归、benchmark 基准。",
            )
            severity = st.radio(
                "severity（严重级别）",
                badcase_manager.SEVERITIES,
                index=2,
                horizontal=True,
                format_func=lambda value: SEVERITY_LABELS.get(value, value),
                help="severity 用于标记问题影响程度。",
            )
            problem_description = st.text_area(
                "问题说明",
                value="",
                placeholder="说明这轮回答哪里错了，例如：能力介绍问题不应该联网检索，也不应该引用无关网页。",
            )

            st.markdown("**行为约束**")
            selected_mode_default = (
                "autonomous"
                if run["config"].get("run_mode") == "自主任务"
                else "normal"
            )
            selected_mode = st.radio(
                "selected_mode（运行模式）",
                badcase_manager.SELECTED_MODES,
                index=badcase_manager.SELECTED_MODES.index(selected_mode_default),
                horizontal=True,
                format_func=lambda value: MODE_LABELS.get(value, value),
                help="selected_mode 表示测试时应该用普通问答还是自主任务模式。",
            )
            expected_mode = st.selectbox(
                "expected_mode（期望运行时）",
                [""] + badcase_manager.EXPECTED_MODES,
                index=0,
                format_func=lambda value: MODE_LABELS.get(value, "不限制"),
                help="expected_mode 用来约束 Agent 应该进入哪种运行时。",
            )
            expected_tools = st.multiselect(
                "expected_tools（期望调用工具）",
                badcase_manager.TOOLS,
                format_func=lambda value: TOOL_LABELS.get(value, value),
                help="expected_tools 表示这类问题必须调用的工具。",
            )
            forbidden_tools = st.multiselect(
                "forbidden_tools（禁止调用工具）",
                badcase_manager.TOOLS,
                format_func=lambda value: TOOL_LABELS.get(value, value),
                help="forbidden_tools 表示这类问题不应该调用的工具。",
            )
            expected_sources = st.multiselect(
                "expected_sources（期望资料来源）",
                badcase_manager.SOURCES,
                format_func=lambda value: SOURCE_LABELS.get(value, value),
                help="expected_sources 表示回答应该使用哪些资料来源。",
            )
            forbidden_sources = st.multiselect(
                "forbidden_sources（禁止资料来源）",
                badcase_manager.SOURCES,
                format_func=lambda value: SOURCE_LABELS.get(value, value),
                help="forbidden_sources 表示回答不应该使用哪些资料来源。",
            )

            st.markdown("**答案约束**")
            required_phrases_text = st.text_input(
                "required_phrases（必须出现词，逗号分隔）",
                help="required_phrases 是答案里必须包含的关键词，例如：上传，联网，RAG。",
            )
            expected_answer_phrases_text = st.text_input(
                "expected_answer_phrases（期望回答词，逗号分隔）",
                help="expected_answer_phrases 用于更明确地要求答案必须包含某些表述。",
            )
            forbidden_answer_phrases_text = st.text_input(
                "forbidden_answer_phrases（禁止回答词，逗号分隔）",
                help="forbidden_answer_phrases 是答案里不能出现的词，例如：搜狐，极简生活，根据现有资料。",
            )
            min_answer_chars = st.number_input(
                "min_answer_chars（最少回答字数）",
                min_value=0,
                max_value=1000,
                value=20,
                step=1,
                help="min_answer_chars 用于避免空回答或过短回答通过测试。",
            )
            success_criteria_text = st.text_area(
                "success_criteria（成功标准，每行一条）",
                value="",
                placeholder="例如：不得引用历史上传资料\n必须直接介绍 Agent 能力",
                help="success_criteria 是人工可读的成功标准，后续可转成规则或 LLM-as-Judge rubric。",
            )
            note = st.text_area(
                "note（备注，不参与规则评估）",
                value="",
                help="note 只给开发者看，不参与自动化 eval。",
            )

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
    result = memory_manager.upsert_memory(candidate)
    if result["ok"]:
        st.session_state.memory_notice = "已保存为长期记忆。"
        st.session_state.dismissed_memory_candidates.add(candidate.get("candidate_id"))
    else:
        st.session_state.memory_notice = "保存失败：" + "；".join(result["errors"])


def dismiss_memory_candidate(candidate_id):
    st.session_state.dismissed_memory_candidates.add(candidate_id)


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
            st.markdown(f"**{index + 1}. {memory_manager.TYPE_LABELS.get(candidate['type'], candidate['type'])}**")
            st.write(candidate["value"])
            st.caption(f"key：{candidate['key']}｜confidence（置信度）：{candidate['confidence']:.2f}｜source（来源）：{candidate['source']}")
            left, right = st.columns([0.18, 0.82])
            with left:
                st.button(
                    "保存",
                    key=f"save_memory_candidate_{candidate['candidate_id']}",
                    on_click=confirm_memory_candidate,
                    args=(candidate,),
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
    global uploaded_files, run_mode, router_mode_label, router_mode, max_autonomous_steps, planner_type_label, planner_type, evaluator_type_label, evaluator_type, memory_enabled, memory_write_mode_label, memory_write_mode, source_strategy_label, source_strategy, retrieval_strategy_label, retrieval_strategy, context_packing_label, context_packing_strategy, chunking_strategy_labels, chunking_strategy, top_k, web_max_results, plan_progress_enabled, streaming_enabled, trace_level, deepseek_model_label, deepseek_model
    st.header("资料")
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
    st.subheader("Agent（智能体）配置")
    run_mode = st.radio(
        "运行模式",
        ["普通问答", "自主任务"],
        help="Agent 是智能体；Tool Agent 是会调用工具完成任务的智能体。",
    )
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

    st.divider()
    st.subheader("Memory（记忆）配置")
    memory_enabled = st.toggle(
        "启用 Memory（长期记忆）",
        value=True,
        help="Memory 会记住用户画像、学习偏好和任务进度；它不同于 RAG 资料库。",
    )
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
        f"active（生效）：{stats['active']}｜archived（归档）：{stats['archived']}｜deleted（删除）：{stats['deleted']}"
    )
    with st.expander("查看 / 删除 Memory（记忆）"):
        for item in memory_manager.load_memories():
            if item.get("status") != memory_manager.MEMORY_STATUS_ACTIVE:
                continue
            st.markdown(f"**{memory_manager.TYPE_LABELS.get(item['type'], item['type'])}**")
            st.write(item["value"])
            st.caption(f"key：{item['key']}｜confidence（置信度）：{item['confidence']:.2f}｜use_count（使用次数）：{item.get('use_count', 0)}")
            st.button(
                "删除这条记忆",
                key=f"delete_memory_{item['id']}",
                on_click=memory_manager.delete_memory,
                args=(item["id"],),
            )
            st.divider()

    st.divider()
    st.subheader("RAG（检索增强生成）配置")
    source_strategy_label = st.radio(
        "资料来源策略",
        list(SOURCE_STRATEGY_LABELS.keys()),
        help="用于观察上传资料、网页资料和自动策略对结果的影响。",
    )
    source_strategy = SOURCE_STRATEGY_LABELS[source_strategy_label]
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

    st.divider()
    st.subheader("可观测性")
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

    st.divider()
    st.caption("Agent（智能体）会自动使用上传资料，并联网补充资料；没有上传资料时，会直接联网收集。")
    st.write("DeepSeek（大模型服务）:", "已配置" if deepseek_key else "未配置")
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
    st.write("通义百炼:", "已配置" if dashscope_key else "未配置")
    reranker_status = "已启用" if agent.ENABLE_RERANKER else "未启用"
    st.write("Reranker（重排序器）:", reranker_status)
    planner_status = "行业主流 Runtime（运行时）雏形"
    st.write("Planner（规划器）:", planner_status)
    st.write("Router（路由器）:", router_mode_label)
    st.write("Source（资料来源）:", source_strategy_label)
    st.write("Retrieval（检索）:", retrieval_strategy_label)
    st.write("Packing（上下文打包）:", context_packing_label)
    st.write("Chunking（切分）:", "、".join(chunking_strategy_labels))
    st.write("Model（模型）:", deepseek_model_label)
    st.write("Evaluator（评估器）:", evaluator_type_label)
    st.write("Streaming（流式输出）:", "已启用" if streaming_enabled else "未启用")

    if "upload_status" in st.session_state and st.session_state.upload_status:
        st.divider()
        st.caption("已入库资料")
        for item in st.session_state.upload_status[-8:]:
            st.write(item)


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


st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
.block-container {padding-top: 1.4rem; max-width: 100%;}
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
</style>
""", unsafe_allow_html=True)

settings_col, workspace_col = st.columns([0.28, 0.72], gap="large")

with settings_col:
    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
    render_settings_panel()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="observe-panel">', unsafe_allow_html=True)
    st.subheader("观察面板")
    st.caption("运行过程、引用来源、badcase 与 Memory 候选会在工作区内跟随回答展示；这里保留当前配置快照。")
    st.write("运行模式：", run_mode)
    st.write("路由：", router_mode_label)
    st.write("资料来源：", source_strategy_label)
    st.write("检索：", retrieval_strategy_label)
    st.write("上下文打包：", context_packing_label)
    st.write("切分：", "、".join(chunking_strategy_labels))
    st.write("模型：", deepseek_model_label)
    st.write("Trace：", trace_level)
    st.markdown('</div>', unsafe_allow_html=True)

with workspace_col:
    st.markdown(
        '<div class="main-header-card"><h1>RAG Agent Pro（检索增强智能体教学版）</h1>'
        '<p>Agent 工作区：对话、执行进度、回答主体、参考来源和反馈闭环集中在这里。</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="workspace-note"><p>当前工作区已移到右侧。左侧保留设置和观察面板，右侧优先展示用户任务和 Agent 产物。</p></div>',
        unsafe_allow_html=True,
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


    prompt = None
    with st.form("workspace_prompt_form", clear_on_submit=True):
        st.markdown('<div class="workspace-composer">', unsafe_allow_html=True)
        prompt_text = st.text_area(
            "输入问题",
            placeholder="输入问题，Agent（智能体）会自动检索上传资料和网络资料",
            label_visibility="collapsed",
            height=82,
        )
        submitted_prompt = st.form_submit_button("发送")
        st.markdown('</div>', unsafe_allow_html=True)
        if submitted_prompt:
            prompt = prompt_text.strip()

    if prompt:
        if not deepseek_key:
            st.error("请先配置 DEEPSEEK_API_KEY。")
            st.stop()

        trace_id = generate_trace_id()
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
                    if memory_enabled:
                        retrieved_memories = memory_manager.retrieve_memories(prompt)
                        memory_context = memory_manager.build_memory_context(retrieved_memories)

                    use_autonomous_mode = False
                    autonomous_route_reason = ""
                    if run_mode == "自主任务":
                        use_autonomous_mode, autonomous_route_reason = call_with_supported_kwargs(
                            autonomous_agent.should_use_autonomous_mode,
                            prompt,
                            router_mode=router_mode,
                        )

                    if run_mode == "自主任务" and use_autonomous_mode:
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
                            metadata_scope={"session_id": st.session_state.rag_session_id},
                            progress_callback=handle_plan_progress,
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
                            metadata_scope={"session_id": st.session_state.rag_session_id},
                            stream_callback=answer_stream_callback,
                            progress_callback=handle_plan_progress,
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
                badcase_run = {
                    "trace_id": trace_id,
                    "user_input": prompt,
                    "actual_answer": result["answer"],
                    "config": build_current_config(),
                    "tools_called": extract_tools_from_steps(result.get("steps", [])),
                    "sources_used": extract_source_types(result.get("sources", [])),
                    "planner_mode": result.get("planner_mode", ""),
                    "memory_used": [item.get("id") for item in retrieved_memories],
                    "run_snapshot": {
                        "trace_id": trace_id,
                        "planner_mode": result.get("planner_mode", ""),
                        "tools_called": extract_tools_from_steps(result.get("steps", [])),
                        "sources_used": extract_source_types(result.get("sources", [])),
                        "memory_used": [item.get("id") for item in retrieved_memories],
                        "steps": compact_steps_for_log(result.get("steps", [])),
                        "sources": compact_sources_for_log(result.get("sources", [])),
                        "answer_preview": str(result.get("answer", ""))[:1200],
                    },
                }
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
                    render_memory_confirmation()

                if trace_level != "隐藏":
                    with st.expander("查看 Agent（智能体）执行步骤"):
                        planner_label = (
                            "Autonomous Runtime（自主任务运行时）"
                            if result.get("planner_mode") == "autonomous_runtime"
                            else
                            "LLM Tool Calling（大模型工具调用）"
                            if result.get("planner_mode") == "llm_tool_calling"
                            else "自主模式回退普通问答"
                            if result.get("planner_mode") == "autonomous_fallback"
                            else "行业主流 Runtime（运行时）雏形"
                            if result.get("planner_mode") == "pro_runtime"
                            else "规则兜底"
                        )
                        st.caption(
                            f"Planner（规划器）来源：{planner_label}｜Router（路由器）：{router_mode_label}｜"
                            f"Source（资料来源）：{source_strategy_label}｜Retrieval（检索）：{retrieval_strategy_label}｜"
                            f"Packing（上下文打包）：{context_packing_label}｜"
                            f"Chunking（切分）：{'、'.join(chunking_strategy_labels)}｜"
                            f"Model（模型）：{deepseek_model_label}｜Evaluator（评估器）：{evaluator_type_label}"
                        )
                        for index, step in enumerate(result.get("steps", []), start=1):
                            status_map = {
                                "success": "成功",
                                "warning": "提示",
                                "failed": "失败",
                            }
                            status = status_map.get(step["status"], step["status"])
                            st.markdown(f"**{index}. {step['name']}**")
                            st.caption(
                                f"工具：{step['tool']} | 状态：{status} | 耗时：{step['elapsed_ms']} ms"
                            )
                            if trace_level == "完整":
                                st.write(step["reason"])
                                st.write(step["summary"])
                            else:
                                st.write(step["summary"])
                            if step.get("error"):
                                st.error(step["error"])
                            st.divider()

                if result.get("planner_mode") == "autonomous_runtime":
                    with st.expander("查看 Autonomous Agent（自主智能体）任务状态"):
                        goal = result.get("goal")
                        if goal:
                            st.markdown("**目标**")
                            st.write(goal.objective)
                            st.caption(f"停止原因：{result.get('stop_reason', '')}")

                        st.markdown("**任务队列**")
                        for task in result.get("tasks", []):
                            st.write(f"{task.id}｜{task.title}｜{task.status}")
                            st.caption(f"依赖：{', '.join(task.depends_on) or '无'}｜预期产物：{task.expected_output}")

                        if result.get("critic_results"):
                            st.markdown("**Critic（批判器）结果**")
                            for critic in result["critic_results"]:
                                status = "通过" if critic["passed"] else "未通过"
                                st.write(f"{critic['task_id']}｜{status}｜分数：{critic['score']}")
                                if critic["issues"]:
                                    st.caption("问题：" + "；".join(critic["issues"]))

                        if result.get("reflections"):
                            st.markdown("**Reflect（反思）补救建议**")
                            for reflection in result["reflections"]:
                                st.write(f"{reflection['task_id']} → {reflection['repair_task_id']}")
                                st.caption("问题：" + "；".join(reflection["issues"]))

                if result["sources"]:
                    with st.expander("查看参考来源"):
                        for index, source in enumerate(result["sources"], start=1):
                            title = source["source"]
                            url = source.get("url", "")
                            source_type = source.get("source_type", "unknown")
                            label = source_label(source)
                            st.markdown(f"**{index}. {title}**")
                            st.markdown(f"`{label}`")
                            st.caption(f"类型：{source_type} | chunk（资料片段）类型：{source.get('chunk_type', 'child')}")
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
                                st.caption(" | ".join(location_parts))
                            st.caption(
                                "融合分："
                                f"{source.get('final_score', 0):.4f} | "
                                f"原始分：{source.get('pre_rerank_score', source.get('final_score', 0)):.4f} | "
                                f"意图：{source.get('query_intent', 'general')} | "
                                f"新鲜度：{source.get('freshness_score', 0):.2f} | "
                                f"答案性：{source.get('answerability_score', 0):.2f} | "
                                f"Rerank（重排序）：{source.get('rerank_status', '未启用')} | "
                                f"Rerank（重排序）分：{source.get('rerank_score', '无')} | "
                                f"向量排名：{source.get('vector_rank', '未召回')} | "
                                f"关键词排名：{source.get('bm25_rank', '未召回')} | "
                                f"上下文顺序：{source.get('context_order', index)}"
                            )
                            if url:
                                st.write(url)
                            st.write(source["document"][:300])
                            st.divider()

            except Exception as e:
                error_answer = f"调用失败：{e}"
                error_run = {
                    "trace_id": trace_id,
                    "user_input": prompt,
                    "actual_answer": error_answer,
                    "config": build_current_config(),
                    "tools_called": [],
                    "sources_used": [],
                    "planner_mode": "error",
                    "memory_used": [],
                    "run_snapshot": {
                        "trace_id": trace_id,
                        "planner_mode": "error",
                        "error": str(e)[:1200],
                        "answer_preview": error_answer,
                    },
                }
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
