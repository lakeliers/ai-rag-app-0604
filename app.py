import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from io import BytesIO
import pypdf
from docx import Document


def get_api_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
    return os.getenv("DEEPSEEK_API_KEY", "")


api_key = get_api_key()
if not api_key:
    st.error("⚠️ 没找到 API Key！请配置环境变量或 Streamlit Secrets")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('shibing624/text2vec-base-chinese')

embedding_model = load_embedding_model()


def read_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    if file_name.endswith((".txt", ".md")):
        return file_bytes.decode("utf-8")
    elif file_name.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text
    elif file_name.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text
    else:
        raise ValueError(f"不支持的文件格式：{file_name}")


def recursive_split(text, chunk_size=300, chunk_overlap=50):
    separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
    def split_recursive(text, separators):
        if len(text) <= chunk_size:
            return [text]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                result = []
                current = ""
                for part in parts:
                    candidate = current + (sep if current else "") + part
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            result.append(current)
                        if len(part) > chunk_size:
                            result.extend(split_recursive(part, separators))
                            current = ""
                        else:
                            current = part
                if current:
                    result.append(current)
                return result
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = split_recursive(text, separators)
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_tail = chunks[i - 1][-chunk_overlap:]
                chunk = prev_tail + chunk
            overlapped.append(chunk)
        return overlapped
    return chunks


def get_embedding(text):
    return embedding_model.encode(text)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantic_search(query, chunks, chunk_embeddings, top_k=3):
    query_emb = get_embedding(query)
    scored = []
    for chunk, emb in zip(chunks, chunk_embeddings):
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


st.title("📚 智能文档助手")
st.write("👈 在左侧上传文档（支持 txt / md / pdf / docx），然后在下面提问")

with st.sidebar:
    st.header("📁 上传文档")
    uploaded_file = st.file_uploader(
        "选择文件",
        type=["txt", "md", "pdf", "docx"]
    )
    if uploaded_file:
        if "current_file" not in st.session_state or \
           st.session_state.current_file != uploaded_file.name:
            try:
                with st.spinner(f"📖 读取 {uploaded_file.name}..."):
                    content = read_file(uploaded_file)
                if not content.strip():
                    st.error("⚠️ 文件内容为空或无法解析")
                    st.stop()
                with st.spinner("📑 智能切分..."):
                    chunks = recursive_split(content, chunk_size=300, chunk_overlap=50)
                with st.spinner(f"🔢 向量化 {len(chunks)} 段..."):
                    chunk_embeddings = [get_embedding(c) for c in chunks]
                st.session_state.chunks = chunks
                st.session_state.chunk_embeddings = chunk_embeddings
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                st.success(f"✅ 已加载 {len(chunks)} 段")
            except Exception as e:
                st.error(f"❌ 处理失败：{e}")
                st.stop()
        st.info(
            f"📄 当前文档\n\n"
            f"**{uploaded_file.name}**\n\n"
            f"切分为 **{len(st.session_state.chunks)}** 段\n\n"
            f"总字符数: **{sum(len(c) for c in st.session_state.chunks)}**"
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("问点啥？")

if prompt:
    if "chunks" not in st.session_state:
        st.warning("⚠️ 请先在左侧上传文档！")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 检索中..."):
            results = semantic_search(
                prompt,
                st.session_state.chunks,
                st.session_state.chunk_embeddings,
                top_k=3
            )
        with st.expander(f"📚 找到 {len(results)} 段相关资料（点击查看）"):
            for i, (sim, chunk) in enumerate(results, 1):
                st.markdown(f"**段{i} | 相似度 {sim:.3f}**")
                st.text(chunk[:200] + ("..." if len(chunk) > 200 else ""))
                st.divider()
        
        context = "\n\n---\n\n".join([chunk for _, chunk in results])
        system_prompt = f"""你是文档问答助手，根据下面的文档片段回答用户问题。

文档片段：
{context}

回答规则：
1. 严格基于上述文档，不编造
2. 如果文档没相关信息，明说"文档中未找到相关内容"
3. 回答简洁，不超过 100 字
4. 不用 markdown
"""
        def stream_response():
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        try:
            full_reply = st.write_stream(stream_response())
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_reply
            })
        except Exception as e:
            st.error(f"❌ 调用失败：{e}")
