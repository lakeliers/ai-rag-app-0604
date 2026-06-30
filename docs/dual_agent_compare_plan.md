# Dual Agent Compare Mode 需求规划

## 目标

在现有教学 Agent 页面中新增“双 Agent 对照模式”。用户可以在同一个页面里同时操作两个 Agent 实例，左右并排观察不同配置带来的回答差异。

本次不做复杂实验看板，不做多 Agent 协作编排，只做“同一套 Agent 工作台复制成两个独立实例”。

## 用户体验

- 页面保留原来的“单 Agent 模式”，避免影响当前线上主链路。
- 新增“单 Agent / 双 Agent 对照”页面模式切换。
- 双 Agent 模式中：
  - 左侧为 Agent A。
  - 右侧为 Agent B。
  - 两侧 UI 结构一致。
  - 两侧可以分别上传文件、选择运行模式、资料来源策略、检索策略、context packing 策略、chunking 策略、模型、memory、安全策略和 trace 展示级别。
  - 两侧分别输入问题和发送。
  - 两侧分别展示对话、执行 trace、资料来源和 badcase 反馈入口。

## 状态隔离要求

双 Agent 模式不能复用单 Agent 模式的全局 session state。每个 Agent 实例需要独立保存：

- `messages`
- `rag_session_id`
- `ingested_uploads`
- `upload_status`
- `last_sources`
- `last_agent_run`
- `pending_memory_candidates`
- `dismissed_memory_candidates`
- `memory_notice`
- 输入框草稿

上传资料入库时必须带不同的 `session_id` metadata，检索时也必须带对应 `session_id`，避免 Agent A 读到 Agent B 的上传资料。

## 后端执行要求

双 Agent 模式复用现有核心能力：

- `agent_runtime.run_agent_pro`
- `autonomous_agent.run_autonomous_agent`
- `memory_manager`
- `permission_gate`
- `rag_agent_core`
- `parsing_layer`

不另起一套 RAG / Tool Agent / Autonomous Agent 逻辑，避免教学模式与真实主链路不一致。

## Eval 更新要求

新增对照模式相关测试 case，重点验证：

- Agent A 和 Agent B 的上传资料隔离。
- Agent A 和 Agent B 的配置隔离。
- 不上传资料的一侧不能引用另一侧上传资料。
- 两侧使用不同资料来源策略时，工具调用路径应不同。

这些 case 主要属于 regression / benchmark。smoke set 只补最小关键链路，避免每次改动成本过高。

## 验收标准

上线前需要完成：

- Python 语法检查通过。
- 双 Agent 页面可打开。
- 单 Agent 原页面不受影响。
- smoke / regression / benchmark 全流程真实 API eval 通过。
- 生成 HTML eval 报告。
- Git 提交并推送到 GitHub。

## 回滚方案

开发前已建立 Git tag：

`backup/pre-dual-agent-compare-20260630-205354`

如果新版本出现不可接受问题，可回滚到该 tag 对应的 commit。
