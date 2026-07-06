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
