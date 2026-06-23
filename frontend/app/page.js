"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronDown,
  FileText,
  Loader2,
  MessageSquarePlus,
  Send,
  Settings2,
  SlidersHorizontal,
  Upload,
  UserRound
} from "lucide-react";
import { getStatus, sendChat, submitBadcase, uploadFiles } from "../lib/api";

const sourceOptions = [
  { label: "自动判断", value: "auto" },
  { label: "仅上传资料", value: "upload_only" },
  { label: "仅联网资料", value: "web_only" },
  { label: "上传 + 联网", value: "upload_and_web" }
];

const retrievalOptions = [
  { label: "仅向量检索", value: "vector_only" },
  { label: "向量 + 关键词", value: "vector_bm25" },
  { label: "混合召回 + 融合排序", value: "vector_bm25_rrf" }
];

const packingOptions = [
  { label: "简单取前几条", value: "simple_topk" },
  { label: "优先上传资料", value: "source_priority" },
  { label: "去重 + 权重排序", value: "weighted" },
  { label: "严格控制上下文长度", value: "strict_budget" }
];

const plannerOptions = [
  { label: "规则规划", value: "rules" },
  { label: "大模型工具规划", value: "llm_tool_calling" },
  { label: "失败回退混合规划", value: "fallback_mixed" }
];

const modelOptions = [
  { label: "DeepSeek Flash", value: "deepseek-chat" },
  { label: "DeepSeek Pro", value: "deepseek-reasoner" }
];

const chunkingOptions = [
  { label: "普通文本", value: "plain" },
  { label: "父子关系", value: "parent_child" },
  { label: "表格专用", value: "table" },
  { label: "摘要 chunk", value: "summary" }
];

const examplesWithoutUpload = [
  "你能做些什么？",
  "RAG 是什么？用产品经理能听懂的话解释",
  "最近 AI Agent 有什么新趋势？"
];

const examplesWithUpload = [
  "总结这份资料的核心观点",
  "找出这份资料里的关键证据和不足",
  "基于资料给我一版行动建议"
];

function createSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `session_${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`;
  }
  return `session_${Date.now()}`;
}

function pillLabel(options, value) {
  return options.find((item) => item.value === value)?.label || value;
}

function sourceTypeLabel(type) {
  if (type === "upload") return "上传资料";
  if (type === "web") return "网页资料";
  if (type === "local") return "基础资料";
  return "资料";
}

function latestVisibleStep(events) {
  const running = [...events].reverse().find((item) => item.status === "running");
  if (running) return { title: "正在执行", step: running };
  const warning = [...events].reverse().find((item) => ["failed", "warning"].includes(item.status));
  if (warning) return { title: "需要注意", step: warning };
  const completed = [...events].reverse().find((item) => ["completed", "skipped"].includes(item.status));
  if (completed) return { title: "最近完成", step: completed };
  return null;
}

export default function Home() {
  const [sessionId] = useState(createSessionId);
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatuses, setUploadStatuses] = useState([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [badcaseTarget, setBadcaseTarget] = useState(null);
  const [badcaseIssue, setBadcaseIssue] = useState("");
  const [badcaseExpected, setBadcaseExpected] = useState("");
  const fileInputRef = useRef(null);

  const [config, setConfig] = useState({
    run_mode: "normal",
    source_strategy: "auto",
    retrieval_strategy: "vector_bm25_rrf",
    context_packing_strategy: "strict_budget",
    chunking_strategy: ["parent_child", "table"],
    router_mode: "rules",
    planner_type: "fallback_mixed",
    evaluator_type: "rules",
    memory_enabled: true,
    top_k: 3,
    web_max_results: 2,
    max_autonomous_steps: 3,
    deepseek_model: "deepseek-chat"
  });

  const examples = uploadStatuses.length ? examplesWithUpload : examplesWithoutUpload;

  const activeSummary = useMemo(() => {
    return [
      config.run_mode === "autonomous" ? "自主任务" : "普通问答",
      pillLabel(sourceOptions, config.source_strategy),
      pillLabel(retrievalOptions, config.retrieval_strategy),
      pillLabel(modelOptions, config.deepseek_model)
    ];
  }, [config]);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setStatus({ deepseek_configured: false, dashscope_configured: false }));
  }, []);

  function updateConfig(key, value) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  function toggleChunking(value) {
    setConfig((current) => {
      const exists = current.chunking_strategy.includes(value);
      const next = exists
        ? current.chunking_strategy.filter((item) => item !== value)
        : [...current.chunking_strategy, value];
      return { ...current, chunking_strategy: next.length ? next : ["parent_child"] };
    });
  }

  async function handleUpload(files) {
    const selectedFiles = Array.from(files || []);
    if (!selectedFiles.length) return;
    setIsUploading(true);
    try {
      const result = await uploadFiles({
        sessionId,
        files: selectedFiles,
        chunkingStrategy: config.chunking_strategy
      });
      setUploadStatuses((current) => [...result.statuses, ...current].slice(0, 8));
    } catch (error) {
      setUploadStatuses((current) => [
        { source: "上传失败", status: "error", message: error.message },
        ...current
      ]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleSubmit(nextQuestion = question) {
    const trimmed = nextQuestion.trim();
    if (!trimmed || isRunning) return;
    const userMessage = { role: "user", content: trimmed };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsRunning(true);

    try {
      const result = await sendChat({ sessionId, question: trimmed, config });
      const assistantMessage = {
        role: "assistant",
        content: result.answer,
        result,
        traceId: result.trace_id
      };
      setMessages((current) => [...current, assistantMessage]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `请求失败：${error.message}`,
          error: true
        }
      ]);
    } finally {
      setIsRunning(false);
    }
  }

  async function handleBadcaseSubmit(event) {
    event.preventDefault();
    if (!badcaseTarget || !badcaseIssue.trim()) return;
    const payload = {
      trace_id: badcaseTarget.traceId || "",
      user_input: findPreviousUserMessage(badcaseTarget),
      actual_answer: badcaseTarget.content,
      issue_summary: badcaseIssue,
      expected_behavior: badcaseExpected,
      sources_used: badcaseTarget.result?.source_types || [],
      save_target: "local"
    };
    const result = await submitBadcase(payload);
    setBadcaseTarget(null);
    setBadcaseIssue("");
    setBadcaseExpected("");
    alert(result.ok === false ? `提交失败：${result.errors?.join("；")}` : `已记录反馈：${result.trace_id || payload.trace_id || "本地记录"}`);
  }

  function findPreviousUserMessage(target) {
    const index = messages.indexOf(target);
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      if (messages[cursor]?.role === "user") return messages[cursor].content;
    }
    return "";
  }

  return (
    <main className="appShell">
      <section className="workspace">
        <header className="hero">
          <div>
            <p className="eyebrow">RAG Agent Pro</p>
            <h1>把资料、联网搜索和 Agent 过程放到一个工作区</h1>
            <p>先提问或上传资料；需要教学实验时，再打开右侧配置。</p>
          </div>
          <div className="statusGroup">
            <span className={status?.deepseek_configured ? "okDot" : "badDot"} />
            <span>{status?.deepseek_configured ? "模型已配置" : "缺少 DeepSeek Key"}</span>
          </div>
        </header>

        <section className="composerCard">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                handleSubmit();
              }
            }}
            placeholder="输入问题，Agent 会自动检索上传资料和网络资料"
          />
          <div className="composerActions">
            <div className="uploadInline">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={(event) => handleUpload(event.target.files)}
              />
              <Upload size={16} />
              <span>{isUploading ? "资料入库中..." : "上传资料"}</span>
            </div>
            <button className="primaryButton" disabled={isRunning || !question.trim()} onClick={() => handleSubmit()}>
              {isRunning ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              发送
            </button>
          </div>
          <div className="quickPrompts">
            {examples.map((item) => (
              <button key={item} onClick={() => handleSubmit(item)} disabled={isRunning}>
                {item}
              </button>
            ))}
          </div>
        </section>

        {uploadStatuses.length > 0 && (
          <section className="uploadState">
            <div className="sectionTitle">
              <FileText size={18} />
              当前资料
            </div>
            <div className="uploadGrid">
              {uploadStatuses.map((item, index) => (
                <div className="uploadItem" key={`${item.source}-${index}`}>
                  <strong>{item.source}</strong>
                  <span>{item.status === "ingested" ? `${item.chunk_count} 块资料已入库` : item.message || "已读取"}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="conversation">
          {messages.length === 0 && (
            <div className="emptyConversation">
              <MessageSquarePlus size={24} />
              <strong>从一个问题开始</strong>
              <span>回答、来源、执行过程和反馈入口会集中显示在这里。</span>
            </div>
          )}

          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="avatar">{message.role === "user" ? <UserRound size={18} /> : <Bot size={18} />}</div>
              <div className="messageBody">
                <div className="messageText">{message.content}</div>
                {message.result && <ResultDetails message={message} onBadcase={() => setBadcaseTarget(message)} />}
              </div>
            </article>
          ))}

          {isRunning && (
            <article className="message assistant">
              <div className="avatar"><Bot size={18} /></div>
              <div className="messageBody">
                <div className="thinking">
                  <Loader2 className="spin" size={16} />
                  Agent 正在检索资料并生成回答...
                </div>
              </div>
            </article>
          )}
        </section>
      </section>

      <aside className="sidePanel">
        <section className="panelCard">
          <div className="sectionTitle">
            <Settings2 size={18} />
            本轮配置
          </div>
          <div className="pillList">
            {activeSummary.map((item) => <span key={item}>{item}</span>)}
          </div>
        </section>

        <section className="panelCard">
          <div className="fieldGroup">
            <label>回答模式</label>
            <div className="segmented">
              <button className={config.run_mode === "normal" ? "selected" : ""} onClick={() => updateConfig("run_mode", "normal")}>普通问答</button>
              <button className={config.run_mode === "autonomous" ? "selected" : ""} onClick={() => updateConfig("run_mode", "autonomous")}>自主任务</button>
            </div>
          </div>

          <SelectField
            label="资料来源"
            value={config.source_strategy}
            options={sourceOptions}
            onChange={(value) => updateConfig("source_strategy", value)}
          />

          <SelectField
            label="模型"
            value={config.deepseek_model}
            options={modelOptions}
            onChange={(value) => updateConfig("deepseek_model", value)}
          />
        </section>

        <section className="panelCard">
          <button className="advancedToggle" onClick={() => setShowAdvanced((value) => !value)}>
            <SlidersHorizontal size={18} />
            教学实验设置
            <ChevronDown className={showAdvanced ? "rotate" : ""} size={18} />
          </button>
          {showAdvanced && (
            <div className="advancedContent">
              <SelectField
                label="检索策略"
                value={config.retrieval_strategy}
                options={retrievalOptions}
                onChange={(value) => updateConfig("retrieval_strategy", value)}
              />
              <SelectField
                label="资料整理方式"
                value={config.context_packing_strategy}
                options={packingOptions}
                onChange={(value) => updateConfig("context_packing_strategy", value)}
              />
              <SelectField
                label="规划方式"
                value={config.planner_type}
                options={plannerOptions}
                onChange={(value) => updateConfig("planner_type", value)}
              />
              <div className="fieldGroup">
                <label>切分策略</label>
                <div className="checkList">
                  {chunkingOptions.map((item) => (
                    <label key={item.value}>
                      <input
                        type="checkbox"
                        checked={config.chunking_strategy.includes(item.value)}
                        onChange={() => toggleChunking(item.value)}
                      />
                      {item.label}
                    </label>
                  ))}
                </div>
              </div>
              <RangeField label="资料条数" value={config.top_k} min={1} max={5} onChange={(value) => updateConfig("top_k", value)} />
              <RangeField label="网页结果数" value={config.web_max_results} min={1} max={5} onChange={(value) => updateConfig("web_max_results", value)} />
              <label className="toggleRow">
                <input
                  type="checkbox"
                  checked={config.memory_enabled}
                  onChange={(event) => updateConfig("memory_enabled", event.target.checked)}
                />
                启用长期记忆
              </label>
            </div>
          )}
        </section>

        <section className="panelCard subtle">
          <strong>系统状态</strong>
          <span>通义百炼：{status?.dashscope_configured ? "已配置" : "未配置"}</span>
          <span>Reranker：{status?.reranker_enabled ? "已启用" : "未启用"}</span>
        </section>
      </aside>

      {badcaseTarget && (
        <div className="modalBackdrop" onClick={() => setBadcaseTarget(null)}>
          <form className="modal" onSubmit={handleBadcaseSubmit} onClick={(event) => event.stopPropagation()}>
            <h2>反馈问题</h2>
            <p>系统会自动带上本轮 Trace ID，开发者可据此复盘完整过程。</p>
            <label>
              哪里不对
              <textarea value={badcaseIssue} onChange={(event) => setBadcaseIssue(event.target.value)} required />
            </label>
            <label>
              你期望它怎么答
              <textarea value={badcaseExpected} onChange={(event) => setBadcaseExpected(event.target.value)} />
            </label>
            <div className="modalActions">
              <button type="button" onClick={() => setBadcaseTarget(null)}>取消</button>
              <button type="submit" className="primaryButton">提交反馈</button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

function ResultDetails({ message, onBadcase }) {
  const result = message.result;
  const latestStep = latestVisibleStep(result.progress_events || []);

  return (
    <div className="resultDetails">
      {latestStep && (
        <div className="progressStrip">
          <span>{latestStep.title}</span>
          <strong>{latestStep.step.name}</strong>
          <em>{latestStep.step.summary}</em>
        </div>
      )}

      {result.sources?.length > 0 && (
        <details className="sourcesBlock" open>
          <summary>参考来源（{result.sources.length}）</summary>
          <div className="sourceCards">
            {result.sources.slice(0, 4).map((source, index) => (
              <div className="sourceCard" key={`${source.source}-${index}`}>
                <div>
                  <strong>{source.source || `资料 ${index + 1}`}</strong>
                  <span>{sourceTypeLabel(source.source_type)}{source.page ? `｜第 ${source.page} 页` : ""}</span>
                </div>
                <p>{source.text || "暂无可展示片段"}</p>
              </div>
            ))}
          </div>
        </details>
      )}

      <details className="debugBlock">
        <summary>过程与调试</summary>
        <div className="traceMeta">Trace ID：{result.trace_id}</div>
        {(result.steps || []).map((step, index) => (
          <div className="traceItem" key={`${step.name}-${index}`}>
            <strong>{step.name}</strong>
            <span>{step.summary || step.reason}</span>
          </div>
        ))}
      </details>

      <button className="feedbackButton" onClick={onBadcase}>
        <AlertTriangle size={15} />
        反馈问题
      </button>
    </div>
  );
}

function SelectField({ label, value, options, onChange }) {
  return (
    <label className="fieldGroup">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((item) => (
          <option value={item.value} key={item.value}>{item.label}</option>
        ))}
      </select>
    </label>
  );
}

function RangeField({ label, value, min, max, onChange }) {
  return (
    <label className="fieldGroup">
      <span>{label}：{value}</span>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
