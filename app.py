import os

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

import rag_agent_core as agent
import parsing_layer
import agent_runtime
import autonomous_agent

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


def file_key(uploaded_file):
    return f"{uploaded_file.name}:{len(uploaded_file.getvalue())}"


def ingest_uploaded_files(uploaded_files, question):
    if not uploaded_files:
        return []

    ingested_sources = []

    for uploaded_file in uploaded_files:
        key = file_key(uploaded_file)
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
            )
        else:
            sections = read_upload_as_sections(uploaded_file)
            source = f"上传：{uploaded_file.name}"
            chunk_count = agent.add_sections_to_chroma(
                sections,
                source=source,
                source_type="upload",
                url=uploaded_file.name,
            )

        st.session_state.ingested_uploads[key] = source
        st.session_state.upload_status.append(f"{source}：{chunk_count} 块")
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
    "自动判断": agent_runtime.SOURCE_STRATEGY_AUTO,
    "仅上传资料": agent_runtime.SOURCE_STRATEGY_UPLOAD_ONLY,
    "仅联网资料": agent_runtime.SOURCE_STRATEGY_WEB_ONLY,
    "上传资料 + 联网并行": agent_runtime.SOURCE_STRATEGY_UPLOAD_AND_WEB,
}


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
        agent_runtime.ROUTER_MODE_HYBRID
        if router_mode_label == "规则-LLM-规则路由"
        else agent_runtime.ROUTER_MODE_RULES
    )
    max_autonomous_steps = st.slider("自主任务最大步数", 1, 5, 3)

    st.divider()
    st.subheader("RAG 配置")
    source_strategy_label = st.radio(
        "资料来源策略",
        list(SOURCE_STRATEGY_LABELS.keys()),
        help="用于观察上传资料、网页资料和自动策略对结果的影响。",
    )
    source_strategy = SOURCE_STRATEGY_LABELS[source_strategy_label]
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


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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
                uploaded_sources = ingest_uploaded_files(uploaded_files, prompt)

                use_autonomous_mode = False
                autonomous_route_reason = ""
                if run_mode == "自主任务":
                    use_autonomous_mode, autonomous_route_reason = autonomous_agent.should_use_autonomous_mode(
                        prompt,
                        router_mode=router_mode,
                    )

                if run_mode == "自主任务" and use_autonomous_mode:
                    result = autonomous_agent.run_autonomous_agent(
                        prompt,
                        top_k=top_k,
                        web_max_results=web_max_results,
                        max_steps=max_autonomous_steps,
                        preferred_sources=uploaded_sources,
                        router_mode=router_mode,
                        source_strategy=source_strategy,
                    )
                else:
                    result = agent_runtime.run_agent_pro(
                        prompt,
                        use_web=True,
                        top_k=top_k,
                        web_max_results=web_max_results,
                        preferred_sources=uploaded_sources,
                        router_mode=router_mode,
                        source_strategy=source_strategy,
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

            st.write(result["answer"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
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
                    st.caption(f"Planner来源：{planner_label}｜Router：{router_mode_label}｜Source：{source_strategy_label}")
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
