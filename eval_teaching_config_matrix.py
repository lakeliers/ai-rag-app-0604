import html
import json
import time
from pathlib import Path
from typing import Any

import agent_runtime
import rag_agent_core as agent
from chunking_layer import ChunkCandidate
from parsing_layer import ParsedSection
from eval_agent_rules import install_fake_tools, restore_tools


ROOT = Path(__file__).resolve().parent
REPORT_HTML = ROOT / "reports" / "teaching_config_matrix_report.html"
REPORT_JSON = ROOT / "reports" / "teaching_config_matrix_report.json"


BASE_CONFIG = {
    "run_mode": "normal",
    "router_mode": "rules",
    "source_strategy": "auto",
    "retrieval_strategy": "vector_bm25_rrf",
    "context_packing_strategy": "strict_budget",
    "chunking_strategy": "parent_child",
    "planner_type": "fallback_mixed",
    "evaluator_type": "rules",
    "reranker_enabled": True,
    "trace_level": "简洁",
    "top_k": 3,
    "web_max_results": 2,
}


CONFIG_CASES = [
    ("运行模式", "普通问答", {"run_mode": "normal"}, "RAG 是什么？", []),
    ("运行模式", "自主任务", {"run_mode": "autonomous"}, "调研 AI Agent 产品趋势，并给产品经理学习建议", []),
    ("路由模式", "规则路由", {"router_mode": "rules"}, "你能做些什么", []),
    ("路由模式", "规则-LLM-规则路由", {"router_mode": "hybrid"}, "你能做些什么", []),
    ("资料来源策略", "自动判断", {"source_strategy": "auto"}, "RAG 是什么？", []),
    ("资料来源策略", "仅上传资料", {"source_strategy": "upload_only"}, "总结这份资料的核心观点", ["上传：AI产品经理学习笔记.md"]),
    ("资料来源策略", "仅联网资料", {"source_strategy": "web_only"}, "最近 AI Agent 有什么新趋势？", []),
    ("资料来源策略", "上传资料 + 联网并行", {"source_strategy": "upload_and_web"}, "结合我上传的资料，再查一下最近有没有类似案例", ["上传：AI产品经理学习笔记.md"]),
    ("检索策略", "仅向量检索", {"retrieval_strategy": "vector_only"}, "BM25 和向量检索有什么区别？", []),
    ("检索策略", "向量 + BM25", {"retrieval_strategy": "vector_bm25"}, "BM25 和向量检索有什么区别？", []),
    ("检索策略", "向量 + BM25 + RRF", {"retrieval_strategy": "vector_bm25_rrf"}, "BM25 和向量检索有什么区别？", []),
    ("Context Packing", "简单 TopK", {"context_packing_strategy": "simple_topk"}, "总结这份资料的核心观点", ["上传：AI产品经理学习笔记.md"]),
    ("Context Packing", "来源优先", {"context_packing_strategy": "source_priority"}, "总结这份资料的核心观点", ["上传：AI产品经理学习笔记.md"]),
    ("Context Packing", "去重 + 新鲜度 + 来源权重", {"context_packing_strategy": "weighted"}, "结合我上传的资料，再查一下最近有没有类似案例", ["上传：AI产品经理学习笔记.md"]),
    ("Context Packing", "严格 token budget", {"context_packing_strategy": "strict_budget"}, "结合我上传的资料，再查一下最近有没有类似案例", ["上传：AI产品经理学习笔记.md"]),
    ("Chunking", "普通文本切分", {"chunking_strategy": "plain"}, "上传同一份长文档后问：第二部分讲了什么？", []),
    ("Chunking", "Parent-child", {"chunking_strategy": "parent_child"}, "上传同一份长文档后问：第二部分讲了什么？", []),
    ("Chunking", "表格专用", {"chunking_strategy": "table"}, "上传 CSV 后问：第 2 行的状态是什么？", []),
    ("Chunking", "摘要 chunk", {"chunking_strategy": "summary"}, "上传长文档后问：全文核心结论是什么？", []),
    ("Planner 类型", "规则 Planner", {"planner_type": "rules"}, "最近 AI Agent 有什么新趋势？", []),
    ("Planner 类型", "LLM Tool Calling Planner", {"planner_type": "llm_tool_calling"}, "最近 AI Agent 有什么新趋势？", []),
    ("Planner 类型", "fallback 混合 Planner", {"planner_type": "fallback_mixed"}, "最近 AI Agent 有什么新趋势？", []),
    ("Evaluator / Critic", "关闭", {"evaluator_type": "off"}, "RAG 是什么？", []),
    ("Evaluator / Critic", "规则评估", {"evaluator_type": "rules"}, "RAG 是什么？", []),
    ("Reranker", "关闭", {"reranker_enabled": False}, "RAG 是什么？", []),
    ("Reranker", "开启", {"reranker_enabled": True}, "RAG 是什么？", []),
    ("Trace 展示级别", "隐藏", {"trace_level": "隐藏"}, "你能做些什么", []),
    ("Trace 展示级别", "简洁", {"trace_level": "简洁"}, "你能做些什么", []),
    ("Trace 展示级别", "完整", {"trace_level": "完整"}, "你能做些什么", []),
    ("资料条数", "TopK=1", {"top_k": 1}, "RAG 是什么？", []),
    ("资料条数", "TopK=5", {"top_k": 5}, "RAG 是什么？", []),
    ("网页结果数", "web=1", {"web_max_results": 1}, "最近 AI Agent 有什么新趋势？", []),
    ("网页结果数", "web=5", {"web_max_results": 5}, "最近 AI Agent 有什么新趋势？", []),
]


def trace_tools(result: dict[str, Any]) -> list[str]:
    return [step.get("tool", "") for step in result.get("steps", [])]


def source_types(result: dict[str, Any]) -> list[str]:
    return sorted({item.get("source_type", "unknown") for item in result.get("sources", [])})


def run_agent_case(config: dict[str, Any], prompt: str, preferred_sources: list[str]) -> dict[str, Any]:
    run_mode = config["run_mode"]
    if run_mode == "autonomous":
        use_autonomous, _ = agent_runtime.classify_intent(prompt, preferred_sources, router_mode=config["router_mode"]), ""
        if use_autonomous.intent in {"chitchat", "capability_intro", "upload_status"}:
            run_mode = "normal"
        else:
            import autonomous_agent

            return autonomous_agent.run_autonomous_agent(
                prompt,
                top_k=config["top_k"],
                web_max_results=config["web_max_results"],
                max_steps=3,
                preferred_sources=preferred_sources,
                router_mode=config["router_mode"],
                source_strategy=config["source_strategy"],
                retrieval_strategy=config["retrieval_strategy"],
                context_packing_strategy=config["context_packing_strategy"],
                planner_type=config["planner_type"],
                evaluator_type=config["evaluator_type"],
            )

    return agent_runtime.run_agent_pro(
        prompt,
        use_web=True,
        top_k=config["top_k"],
        web_max_results=config["web_max_results"],
        preferred_sources=preferred_sources,
        router_mode=config["router_mode"],
        source_strategy=config["source_strategy"],
        retrieval_strategy=config["retrieval_strategy"],
        context_packing_strategy=config["context_packing_strategy"],
        planner_type=config["planner_type"],
        evaluator_type=config["evaluator_type"],
    )


def evaluate_chunking(strategy: str) -> dict[str, Any]:
    text_section = ParsedSection(
        text="第一部分：Agent 需要先识别目标。\n\n第二部分：RAG 需要先解析、切分、入库，再检索和生成。\n\n第三部分：Eval 用来保证改动不破坏主链路。",
        section_title="Agent 教学材料",
        content_type="text",
    )
    table_section = ParsedSection(
        text="姓名 | 状态 | 分数\nA | 通过 | 90\nB | 待确认 | 70\nC | 失败 | 40",
        section_title="测试表格",
        content_type="table",
        sheet="Sheet1",
        row_start=1,
        row_end=4,
    )
    section = table_section if strategy == "table" else text_section
    chunks: list[ChunkCandidate] = agent.build_chunk_candidates(section, "教学配置测试", 1, strategy)
    if strategy == "summary":
        summary = agent.summarize_section_for_retrieval(section, force=True)
        if summary:
            chunks.insert(0, ChunkCandidate(text=f"摘要：{summary}", chunk_type="summary"))

    return {
        "chunk_count": len(chunks),
        "chunk_types": sorted({chunk.chunk_type for chunk in chunks}),
        "sample": chunks[0].text[:160] if chunks else "",
    }


def run_matrix() -> dict[str, Any]:
    original_tools = install_fake_tools()
    original_llm_planner = agent_runtime.ENABLE_LLM_PLANNER
    original_reranker = agent.ENABLE_RERANKER
    agent_runtime.ENABLE_LLM_PLANNER = False
    rows = []
    try:
        for group, option, overrides, prompt, preferred_sources in CONFIG_CASES:
            config = {**BASE_CONFIG, **overrides}
            started_at = time.time()
            error = ""
            result: dict[str, Any] = {}
            extra: dict[str, Any] = {}
            try:
                agent.ENABLE_RERANKER = config["reranker_enabled"]
                if group == "Chunking":
                    extra = evaluate_chunking(config["chunking_strategy"])
                    result = {
                        "planner_mode": "chunking_eval",
                        "answer": extra["sample"],
                        "sources": [],
                        "steps": [],
                    }
                else:
                    result = run_agent_case(config, prompt, preferred_sources)
                passed = bool(result.get("answer")) or bool(extra.get("chunk_count"))
            except Exception as exc:
                passed = False
                error = str(exc)

            rows.append({
                "group": group,
                "option": option,
                "passed": passed,
                "prompt": prompt,
                "config": config,
                "planner_mode": result.get("planner_mode", ""),
                "tools": trace_tools(result),
                "source_types": source_types(result),
                "answer_preview": str(result.get("answer", ""))[:220],
                "extra": extra,
                "elapsed_ms": int((time.time() - started_at) * 1000),
                "error": error,
            })
    finally:
        restore_tools(original_tools)
        agent_runtime.ENABLE_LLM_PLANNER = original_llm_planner
        agent.ENABLE_RERANKER = original_reranker

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "mock-plumbing",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0,
        "rows": rows,
    }


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(report: dict[str, Any]) -> None:
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for row in report["rows"]:
        status = "通过" if row["passed"] else "失败"
        status_class = "pass" if row["passed"] else "fail"
        rows_html.append(f"""
        <tr>
          <td>{esc(row['group'])}</td>
          <td>{esc(row['option'])}</td>
          <td class="{status_class}">{status}</td>
          <td>{esc(row['prompt'])}</td>
          <td><pre>{esc(json.dumps(row['config'], ensure_ascii=False, indent=2))}</pre></td>
          <td><pre>{esc(json.dumps({
              "planner_mode": row["planner_mode"],
              "tools": row["tools"],
              "source_types": row["source_types"],
              "extra": row["extra"],
              "answer_preview": row["answer_preview"],
              "elapsed_ms": row["elapsed_ms"],
              "error": row["error"],
          }, ensure_ascii=False, indent=2))}</pre></td>
        </tr>
        """)

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Teaching Config Matrix Eval</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .sub {{ color: #6b7280; margin-bottom: 24px; }}
    .metrics {{ display: flex; gap: 12px; margin-bottom: 24px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; min-width: 120px; background: #f9fafb; }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #b91c1c; font-weight: 700; }}
    pre {{ margin: 0; white-space: pre-wrap; font-size: 12px; max-height: 260px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Teaching Config Matrix Eval</h1>
  <div class="sub">生成时间：{esc(report['generated_at'])}。模式：{esc(report['mode'])}。用于检查每个教学配置项是否能正常跑通。</div>
  <div class="metrics">
    <div class="card"><div class="metric">{report['total']}</div><div>配置样本</div></div>
    <div class="card"><div class="metric pass">{report['passed']}</div><div>通过</div></div>
    <div class="card"><div class="metric fail">{report['failed']}</div><div>失败</div></div>
    <div class="card"><div class="metric">{report['pass_rate']:.0%}</div><div>通过率</div></div>
  </div>
  <table>
    <thead>
      <tr><th>配置组</th><th>选项</th><th>结果</th><th>Prompt</th><th>配置</th><th>运行摘要</th></tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")


def main() -> None:
    report = run_matrix()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    render_html(report)
    print(f"Total: {report['total']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Pass rate: {report['pass_rate']:.0%}")
    print(f"HTML: {REPORT_HTML}")
    print(f"JSON: {REPORT_JSON}")


if __name__ == "__main__":
    main()
