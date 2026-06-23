# RAG Agent Pro Next.js Frontend

这是 React / Next.js 版前端，用于替代 Streamlit 的主要用户交互体验。

## 本地启动

先启动 Python Agent API：

```bash
uvicorn api_server:app --reload --port 8000
```

再启动 Next.js：

```bash
cd frontend
npm install
npm run dev
```

默认前端会请求：

```text
http://127.0.0.1:8000
```

如需修改 API 地址：

```bash
NEXT_PUBLIC_AGENT_API=http://127.0.0.1:8000 npm run dev
```

## 当前覆盖能力

- 上传文件并写入原有 Chroma / RAG 资料库
- 普通问答 / 自主任务模式切换
- 资料来源、检索策略、上下文打包、切分策略、Planner、Memory 等教学配置
- 展示回答、参考来源、执行过程、Trace ID
- 反馈问题并进入 badcase / regression 闭环

## 架构说明

Next.js 负责前端体验；`api_server.py` 负责复用原 Python Agent 能力。

Streamlit 版 `app.py` 暂时保留，便于对比教学和回滚。
