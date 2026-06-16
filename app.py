import os
import inspect

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

agent.seed_local_note()


st.set_page_config(
    page_title="RAG Agent Pro",
    page_icon="🤖",
    layout="wide",
)

st.title("RAG Agent Pro")



def read_upload_as_sections(uploaded_file):
    return parsing_layer.read_upload_as_sections(uploaded_file)

def is_image(uploaded_file):
    return uploaded_file.type.startswith("image/")


def file_key(uploaded_file, chunking_strategy):
    return f"{uploaded_file.name}:{len(uploaded_file.getvalue())}:{chunking_strategy}"


def ingest_uploaded_files(uploaded_files, question, chunking_strategy):
    if not uploaded_files:
        return []

    ingested_sources = []

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
            )

        st.session_state.ingested_uploads[key] = source
        st.session_state.upload_status.append(f"{source}：{chunk_count} 块｜切分：{chunking_strategy}")
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
    "简单 TopK": "simple_topk",
    "来源优先": "source_priority",
    "去重 + 新鲜度 + 来源权重": "weighted",
    "严格 token budget": "strict_budget",
}
CHUNKING_STRATEGY_LABELS = {
    "普通文本切分": "plain",
    "Parent-child": "parent_child",
    "表格专用": "table",
    "摘要 chunk": "summary",
}
PLANNER_TYPE_LABELS = {
    "规则 Planner": "rules",
    "LLM Tool Calling Planner": "llm_tool_calling",
    "fallback 混合 Planner": "fallback_mixed",
}
EVALUATOR_TYPE_LABELS = {
    "关闭": "off",
    "规则评估": "rules",
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


def build_current_config():
    return {
        "run_mode": run_mode,
        "router_mode": router_mode,
        "source_strategy": source_strategy,
        "retrieval_strategy": retrieval_strategy,
        "context_packing_strategy": context_packing_strategy,
        "chunking_strategy": chunking_strategy,
        "planner_type": planner_type,
        "evaluator_type": evaluator_type,
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
    if run:
        with right:
            st.button(
                "!",
                key=f"badcase_button_{key_suffix}",
                help="反馈 badcase",
                on_click=set_badcase_target,
                args=(run,),
            )


def render_badcase_form():
    run = st.session_state.get("last_agent_run")
    if not run or not st.session_state.get("show_badcase_form"):
        return

    with st.expander("反馈 badcase：补充 Regression Case 信息", expanded=True):
        st.markdown("**当前问题现场**")
        st.write("User Prompt：", run["user_input"])
        st.write("Agent Answer：", run["actual_answer"])
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
                help="本地 eval 会直接写入 eval_cases.jsonl；线上 eval 会创建 GitHub Issue 等待开发者确认。",
            )

            st.markdown("**Case 基础信息**")
            category = st.selectbox(
                "category",
                badcase_manager.CATEGORIES,
                index=badcase_manager.CATEGORIES.index(default_category),
            )
            case_id = st.text_input(
                "case_id",
                value=badcase_manager.generate_case_id(run["user_input"], category),
            )
            suite = st.multiselect(
                "suite",
                badcase_manager.SUITES,
                default=["regression"],
            )
            severity = st.radio(
                "severity",
                badcase_manager.SEVERITIES,
                index=2,
                horizontal=True,
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
                "selected_mode",
                badcase_manager.SELECTED_MODES,
                index=badcase_manager.SELECTED_MODES.index(selected_mode_default),
                horizontal=True,
            )
            expected_mode = st.selectbox(
                "expected_mode",
                [""] + badcase_manager.EXPECTED_MODES,
                index=0,
            )
            expected_tools = st.multiselect("expected_tools", badcase_manager.TOOLS)
            forbidden_tools = st.multiselect("forbidden_tools", badcase_manager.TOOLS)
            expected_sources = st.multiselect("expected_sources", badcase_manager.SOURCES)
            forbidden_sources = st.multiselect("forbidden_sources", badcase_manager.SOURCES)

            st.markdown("**答案约束**")
            required_phrases_text = st.text_input(
                "required_phrases（逗号分隔）",
                help="例如：上传，联网，RAG",
            )
            expected_answer_phrases_text = st.text_input(
                "expected_answer_phrases（逗号分隔）",
                help="用于更明确地要求答案必须包含某些表述。",
            )
            forbidden_answer_phrases_text = st.text_input(
                "forbidden_answer_phrases（逗号分隔）",
                help="例如：搜狐，极简生活，根据现有资料",
            )
            min_answer_chars = st.number_input(
                "min_answer_chars",
                min_value=0,
                max_value=1000,
                value=20,
                step=1,
            )
            success_criteria_text = st.text_area(
                "success_criteria（每行一条）",
                value="",
                placeholder="例如：不得引用历史上传资料\n必须直接介绍 Agent 能力",
            )
            note = st.text_area("note（人工备注，不参与规则评估）", value="")

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
                    severity=severity,
                    problem_description=problem_description,
                    note=note,
                )
                if save_result["errors"]:
                    for error in save_result["errors"]:
                        st.error(error)
                else:
                    if save_result["local_eval_saved"]:
                        st.success("已写入本地 eval_cases.jsonl。")
                    if save_result["local_badcase_saved"]:
                        st.success("已写入 bad_cases.jsonl。")
                    if save_result["github_issue_url"]:
                        st.success("已创建 GitHub Issue。")
                        st.write(save_result["github_issue_url"])
            except Exception as error:
                st.error(str(error))


with st.sidebar:
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
    st.subheader("Agent 配置")
    run_mode = st.radio(
        "运行模式",
        ["普通问答", "自主任务"],
        help="普通问答走 Tool Agent；自主任务会先拆任务，再逐步调用 Tool Agent 推进。",
    )
    router_mode_label = st.radio(
        "路由模式",
        ["规则路由", "规则-LLM-规则路由"],
        help="规则路由更快更稳定；规则-LLM-规则路由会在规则不确定时调用模型做语义分类，再由规则复核。",
    )
    router_mode = (
        "hybrid"
        if router_mode_label == "规则-LLM-规则路由"
        else "rules"
    )
    max_autonomous_steps = st.slider("自主任务最大步数", 1, 5, 3)
    planner_type_label = st.selectbox(
        "Planner 类型",
        list(PLANNER_TYPE_LABELS.keys()),
        index=2,
        help="规则 Planner 更稳定；LLM Tool Calling Planner 会让模型选择工具；fallback 混合 Planner 是当前教学默认链路。",
    )
    planner_type = PLANNER_TYPE_LABELS[planner_type_label]
    evaluator_type_label = st.selectbox(
        "Evaluator / Critic",
        list(EVALUATOR_TYPE_LABELS.keys()),
        index=1,
        help="控制 C 端回答链路中的资料充分性判断。关闭会跳过评估；规则评估会检查资料数量、来源和引用可用性。",
    )
    evaluator_type = EVALUATOR_TYPE_LABELS[evaluator_type_label]

    st.divider()
    st.subheader("RAG 配置")
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
        help="用于对比向量召回、关键词召回和 RRF 融合召回的差异。",
    )
    retrieval_strategy = RETRIEVAL_STRATEGY_LABELS[retrieval_strategy_label]
    context_packing_label = st.selectbox(
        "Context Packing 策略",
        list(CONTEXT_PACKING_LABELS.keys()),
        index=3,
        help="控制最终送进大模型的资料如何筛选、去重、配额和预算约束。",
    )
    context_packing_strategy = CONTEXT_PACKING_LABELS[context_packing_label]
    chunking_strategy_label = st.selectbox(
        "Chunking 策略",
        list(CHUNKING_STRATEGY_LABELS.keys()),
        index=1,
        help="对新上传文件入库生效；已入库资料不会自动重切。",
    )
    chunking_strategy = CHUNKING_STRATEGY_LABELS[chunking_strategy_label]
    top_k = st.slider("资料条数", 1, 5, 3)
    web_max_results = st.slider("网页结果数", 1, 5, 2)
    reranker_enabled = st.toggle(
        "启用 Reranker",
        value=agent.ENABLE_RERANKER,
        help="关闭后只使用向量、关键词和融合分；开启后增加精排模型。",
    )
    agent.ENABLE_RERANKER = reranker_enabled

    st.divider()
    st.subheader("可观测性")
    trace_level = st.radio(
        "Trace 展示级别",
        ["简洁", "完整", "隐藏"],
        help="控制是否展示 Agent 执行步骤、工具、原因和耗时。",
    )

    st.divider()
    st.caption("Agent 会自动使用上传资料，并联网补充资料；没有上传资料时，会直接联网收集。")
    st.write("DeepSeek:", "已配置" if deepseek_key else "未配置")
    st.write("通义百炼:", "已配置" if dashscope_key else "未配置")
    reranker_status = "已启用" if agent.ENABLE_RERANKER else "未启用"
    st.write("Reranker:", reranker_status)
    planner_status = "行业主流Runtime雏形"
    st.write("Planner:", planner_status)
    st.write("Router:", router_mode_label)
    st.write("Source:", source_strategy_label)
    st.write("Retrieval:", retrieval_strategy_label)
    st.write("Packing:", context_packing_label)
    st.write("Chunking:", chunking_strategy_label)
    st.write("Evaluator:", evaluator_type_label)

    if "upload_status" in st.session_state and st.session_state.upload_status:
        st.divider()
        st.caption("已入库资料")
        for item in st.session_state.upload_status[-8:]:
            st.write(item)


if "messages" not in st.session_state:
    st.session_state.messages = []

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


prompt = st.chat_input("输入问题，Agent 会自动检索上传资料和网络资料")

if prompt:
    if not deepseek_key:
        st.error("请先配置 DEEPSEEK_API_KEY。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("执行 Agent 计划中..."):
                uploaded_sources = ingest_uploaded_files(uploaded_files, prompt, chunking_strategy)

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
                    )
                else:
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
                    )
                    if run_mode == "自主任务":
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

            badcase_run = {
                "user_input": prompt,
                "actual_answer": result["answer"],
                "config": build_current_config(),
                "tools_called": extract_tools_from_steps(result.get("steps", [])),
                "sources_used": extract_source_types(result.get("sources", [])),
                "planner_mode": result.get("planner_mode", ""),
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

            if trace_level != "隐藏":
                with st.expander("查看 Agent 执行步骤"):
                    planner_label = (
                        "Autonomous Runtime"
                        if result.get("planner_mode") == "autonomous_runtime"
                        else
                        "LLM Tool Calling"
                        if result.get("planner_mode") == "llm_tool_calling"
                        else "自主模式回退普通问答"
                        if result.get("planner_mode") == "autonomous_fallback"
                        else "行业主流Runtime雏形"
                        if result.get("planner_mode") == "pro_runtime"
                        else "规则兜底"
                    )
                    st.caption(
                        f"Planner来源：{planner_label}｜Router：{router_mode_label}｜"
                        f"Source：{source_strategy_label}｜Retrieval：{retrieval_strategy_label}｜"
                        f"Packing：{context_packing_label}｜Evaluator：{evaluator_type_label}"
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
                with st.expander("查看自主任务状态"):
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
                        st.markdown("**Critic 结果**")
                        for critic in result["critic_results"]:
                            status = "通过" if critic["passed"] else "未通过"
                            st.write(f"{critic['task_id']}｜{status}｜分数：{critic['score']}")
                            if critic["issues"]:
                                st.caption("问题：" + "；".join(critic["issues"]))

                    if result.get("reflections"):
                        st.markdown("**Reflect 补救建议**")
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
                        st.caption(f"类型：{source_type} | 块类型：{source.get('chunk_type', 'child')}")
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
                            f"Rerank：{source.get('rerank_status', '未启用')} | "
                            f"Rerank分：{source.get('rerank_score', '无')} | "
                            f"向量排名：{source.get('vector_rank', '未召回')} | "
                            f"关键词排名：{source.get('bm25_rank', '未召回')} | "
                            f"上下文顺序：{source.get('context_order', index)}"
                        )
                        if url:
                            st.write(url)
                        st.write(source["document"][:300])
                        st.divider()

        except Exception as e:
            st.error(f"调用失败：{e}")

render_badcase_form()
