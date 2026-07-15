# agent for train

当前主线版本是 Streamlit 应用，线上 Streamlit Cloud 使用 `app.py` 作为生产入口。

## 生产入口

```bash
streamlit run app.py
```

线上地址：

```text
https://agent-for-train.streamlit.app/
```

## 云端 Embedding

生产环境默认使用阿里云百炼的 `text-embedding-v4`，以 1024 维向量写入独立的
Chroma 集合 `file_docs_dashscope_text_embedding_v4_1024`。旧集合 `file_docs`
会保留，不会与新模型生成的向量混用。

Streamlit Secrets 必填：

```toml
DASHSCOPE_API_KEY = "你的真实 Key"
```

以下配置均为可选，未填写时使用右侧默认值：

```toml
EMBEDDING_PROVIDER = "dashscope"
EMBEDDING_MODEL_NAME = "text-embedding-v4"
EMBEDDING_DIMENSIONS = "1024"
```

云端向量服务异常时，上传入库会明确提示失败；混合检索会记录降级状态并暂时使用
BM25，不会临时切换到另一套向量模型，以免查询向量与库存量不兼容。

## 实验入口：React / Next.js

`frontend/` 和 `api_server.py` 是产品化前端实验版本，用于后续对比教学和用户体验验证。它们目前不是线上 Streamlit Cloud 的生产入口。

本地体验实验版：

```bash
./dev_next.sh
```

然后打开：

```text
http://localhost:3000
```

## 当前取舍

- Streamlit：继续作为主线，适合快速教学、RAG/Agent 能力实验和 Streamlit Cloud 部署。
- Next.js：保留为实验分支方向，适合后续产品级 UI，但需要单独部署前端和 Python API。
