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

if deepseek_key:
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
if dashscope_key:
    os.environ["DASHSCOPE_API_KEY"] = dashscope_key

import rag_agent_core as agent

agent.seed_local_note()


st.set_page_config(
    page_title="RAG Agent Pro",
    page_icon="🤖",
    layout="wide",
)

st.title("RAG Agent Pro")

with st.sidebar:
    st.header("设置")
    use_web = st.checkbox("联网检索", value=False)
    use_image = st.checkbox("使用图片", value=False)
    uploaded_image = st.file_uploader(
        "上传图片",
        type=["png", "jpg", "jpeg", "webp"],
        disabled=not use_image,
    )

    top_k = st.slider("资料条数", 1, 5, 3)
    web_max_results = st.slider("网页结果数", 1, 5, 3)

    st.divider()
    st.caption("Key 从 Streamlit Secrets 或环境变量读取。")
    st.write("DeepSeek:", "已配置" if deepseek_key else "未配置")
    st.write("通义百炼:", "已配置" if dashscope_key else "未配置")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("输入问题，例如：Chroma 是什么？")

if prompt:
    if not deepseek_key:
        st.error("请先配置 DEEPSEEK_API_KEY。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("检索和思考中..."):
                question = prompt

                if use_image:
                    if not dashscope_key:
                        st.error("使用图片需要配置 DASHSCOPE_API_KEY。")
                        st.stop()
                    if uploaded_image is None:
                        st.error("请先在左侧上传图片。")
                        st.stop()

                    image_summary = agent.describe_image_bytes(
                        uploaded_image.getvalue(),
                        uploaded_image.type,
                        prompt,
                    )
                    agent.add_text_to_chroma(
                        image_summary,
                        source=f"图片：{uploaded_image.name}",
                        source_type="image",
                        url=uploaded_image.name,
                    )

                if prompt.startswith("/web "):
                    question = prompt[len("/web "):].strip()
                    use_web_now = True
                else:
                    use_web_now = use_web

                result = agent.answer_with_rag(
                    question,
                    use_web=use_web_now,
                    top_k=top_k,
                    web_max_results=web_max_results,
                )

            st.write(result["answer"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
            })
            st.session_state.last_sources = result["sources"]

            with st.expander("查看参考来源"):
                for index, source in enumerate(result["sources"], start=1):
                    title = source["source"]
                    url = source.get("url", "")
                    source_type = source.get("source_type", "unknown")
                    st.markdown(f"**{index}. {title}**")
                    st.caption(f"类型：{source_type}")
                    if url:
                        st.write(url)
                    st.write(source["document"][:300])
                    st.divider()

        except Exception as e:
            st.error(f"调用失败：{e}")
