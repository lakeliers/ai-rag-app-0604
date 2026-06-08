import html
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

import rag_agent_core as agent


MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
REPORT_PATH = Path("reports/rag_pipeline_ab_report.html")


@dataclass
class EvalRow:
    id: str
    source: str
    source_type: str
    document: str
    vector_rank: int | None
    bm25_rank: int | None
    created_at: int
    document_key: str
    chunk_index: int = 0
    url: str = ""


@dataclass
class EvalCase:
    name: str
    question: str
    top_k: int
    rows: list[EvalRow]
    expected_source: str
    expected_text: str


NOW = int(time.time())
DAY = 24 * 60 * 60


CASES = [
    EvalCase(
        name="latest_news_prefers_fresh_web",
        question="最近 AI Agent 有什么新趋势？",
        top_k=3,
        rows=[
            EvalRow(
                id="old_local_ai_agent",
                source="基础资料：AI Agent 入门笔记",
                source_type="local",
                document="AI Agent 是能够调用工具、规划任务并执行多步骤操作的人工智能系统。",
                vector_rank=1,
                bm25_rank=1,
                created_at=NOW - 500 * DAY,
                document_key="local:ai_agent_note",
            ),
            EvalRow(
                id="fresh_web_trend",
                source="网页：2026 AI Agent 趋势报告",
                source_type="web",
                document="2026 年 AI Agent 的新趋势包括多智能体协作、浏览器操作、企业流程自动化和可评估 Agent 工作流。",
                vector_rank=3,
                bm25_rank=2,
                created_at=NOW - 2 * DAY,
                document_key="web:agent_trend_2026",
                url="https://example.com/agent-trend",
            ),
            EvalRow(
                id="web_noise",
                source="网页：AI 新闻导航页",
                source_type="web",
                document="本页面聚合 AI 新闻、工具、教程和热门产品列表。",
                vector_rank=2,
                bm25_rank=4,
                created_at=NOW - 1 * DAY,
                document_key="web:ai_nav",
                url="https://example.com/ai-news",
            ),
        ],
        expected_source="网页：2026 AI Agent 趋势报告",
        expected_text="多智能体协作",
    ),
    EvalCase(
        name="dedupe_limits_same_document",
        question="客服升级后的目标响应时间是多少？",
        top_k=3,
        rows=[
            EvalRow(
                id="prd_chunk_1",
                source="上传：客服改版需求.docx",
                source_type="upload",
                document="客服升级目标：首响时间控制在 30 秒以内，优先处理会员用户咨询。",
                vector_rank=1,
                bm25_rank=1,
                created_at=NOW - 1 * DAY,
                document_key="upload:cs_prd",
            ),
            EvalRow(
                id="prd_chunk_2",
                source="上传：客服改版需求.docx",
                source_type="upload",
                document="客服升级目标：首响时间控制在 30 秒以内，优先处理会员用户咨询。",
                vector_rank=2,
                bm25_rank=2,
                created_at=NOW - 1 * DAY,
                document_key="upload:cs_prd",
                chunk_index=1,
            ),
            EvalRow(
                id="prd_chunk_3",
                source="上传：客服改版需求.docx",
                source_type="upload",
                document="升级后的客服工作台需要展示用户历史订单、会员等级和最近一次投诉。",
                vector_rank=3,
                bm25_rank=3,
                created_at=NOW - 1 * DAY,
                document_key="upload:cs_prd",
                chunk_index=2,
            ),
            EvalRow(
                id="web_industry",
                source="网页：客服行业文章",
                source_type="web",
                document="行业客服响应时间通常要求在 5 分钟以内，但不同业务会有不同 SLA。",
                vector_rank=4,
                bm25_rank=4,
                created_at=NOW - 5 * DAY,
                document_key="web:cs_sla",
            ),
        ],
        expected_source="上传：客服改版需求.docx",
        expected_text="30 秒",
    ),
    EvalCase(
        name="policy_or_prd_prefers_upload",
        question="这个 PRD 的上线时间和价格是什么？",
        top_k=3,
        rows=[
            EvalRow(
                id="web_price",
                source="网页：外部促销报道",
                source_type="web",
                document="公开报道显示，该产品预计 6 月 20 日上线，会员首月价格为 199 元。",
                vector_rank=1,
                bm25_rank=1,
                created_at=NOW - 1 * DAY,
                document_key="web:promo",
            ),
            EvalRow(
                id="upload_prd",
                source="上传：产品 PRD.md",
                source_type="upload",
                document="内部 PRD 明确写着：产品上线时间为 6 月 10 日，会员首月价格为 129 元。",
                vector_rank=2,
                bm25_rank=2,
                created_at=NOW - 2 * DAY,
                document_key="upload:product_prd",
            ),
            EvalRow(
                id="local_note",
                source="基础资料：学习笔记",
                source_type="local",
                document="RAG 是检索增强生成，Chroma 是向量数据库。",
                vector_rank=3,
                bm25_rank=None,
                created_at=NOW - 300 * DAY,
                document_key="local:note",
            ),
        ],
        expected_source="上传：产品 PRD.md",
        expected_text="129",
    ),
    EvalCase(
        name="definition_ignores_freshness",
        question="RAG 是什么？",
        top_k=3,
        rows=[
            EvalRow(
                id="fresh_web_short",
                source="网页：今日 AI 快讯",
                source_type="web",
                document="今日 AI 快讯提到 RAG、Agent、模型压缩等热门话题。",
                vector_rank=1,
                bm25_rank=2,
                created_at=NOW,
                document_key="web:daily",
            ),
            EvalRow(
                id="local_definition",
                source="基础资料：RAG 学习笔记",
                source_type="local",
                document="RAG（Retrieval-Augmented Generation，检索增强生成）是在回答前先从知识库检索相关资料，再结合大模型生成答案的技术。",
                vector_rank=2,
                bm25_rank=1,
                created_at=NOW - 400 * DAY,
                document_key="local:rag_note",
            ),
            EvalRow(
                id="web_noise",
                source="网页：AI 工具榜单",
                source_type="web",
                document="这里整理了热门 AI 工具榜单和教程入口。",
                vector_rank=3,
                bm25_rank=None,
                created_at=NOW - 1 * DAY,
                document_key="web:tools",
            ),
        ],
        expected_source="基础资料：RAG 学习笔记",
        expected_text="检索增强生成",
    ),
]


def as_search_row(row: EvalRow):
    item = {
        "id": row.id,
        "source": row.source,
        "source_type": row.source_type,
        "url": row.url,
        "document": row.document,
        "chunk_index": row.chunk_index,
        "created_at": row.created_at,
        "content_type": "text",
        "document_key": row.document_key,
        "distance": None,
        "bm25_score": 0.0,
        "vector_rank": row.vector_rank,
        "bm25_rank": row.bm25_rank,
    }
    rrf_score = 0.0
    if row.vector_rank:
        rrf_score += 1 / (agent.RRF_K + row.vector_rank)
    if row.bm25_rank:
        rrf_score += 1 / (agent.RRF_K + row.bm25_rank)
    item["rrf_score"] = rrf_score
    return item


def old_pipeline(case: EvalCase):
    keywords = agent.extract_query_keywords(case.question)
    rows = []
    for eval_row in case.rows:
        row = as_search_row(eval_row)
        source_priority = 1 if row["source_type"] == "upload" else 0
        keyword_hits = agent.keyword_score(row["document"], keywords)
        row["keyword_score"] = keyword_hits
        row["source_priority"] = source_priority
        row["source_weight"] = agent.source_weight(row["source_type"])
        row["freshness_score"] = agent.freshness_score(row)
        row["query_intent"] = "fixed_old_strategy"
        row["rerank_status"] = "未启用"
        row["final_score"] = (
            row.get("rrf_score", 0)
            * agent.source_weight(row["source_type"])
            * (1.2 if source_priority else 1.0)
            + keyword_hits * 0.01
        )
        row["pre_rerank_score"] = row["final_score"]
        rows.append(row)

    rows.sort(key=lambda item: item["final_score"], reverse=True)
    selected = agent.apply_source_quotas(rows, case.top_k)
    for index, row in enumerate(selected, start=1):
        row["context_order"] = index
    return selected


def new_pipeline(case: EvalCase):
    query_profile = agent.analyze_query(case.question)
    keywords = agent.extract_query_keywords(case.question)
    query_profile["keywords"] = keywords
    rows = []

    for eval_row in case.rows:
        row = as_search_row(eval_row)
        source_priority = 1 if row["source_type"] == "upload" else 0
        keyword_hits = agent.keyword_score(row["document"], keywords)
        row["keyword_score"] = keyword_hits
        row["source_priority"] = source_priority
        row["source_weight"] = agent.source_weight(row["source_type"])
        row["freshness_score"] = agent.freshness_score(row)
        row["query_intent"] = query_profile["intent"]
        row["ranking_weights"] = str(query_profile["weights"])
        row["rerank_status"] = "未启用"
        row["final_score"] = agent.base_retrieval_score(
            row,
            source_priority,
            keyword_hits,
            query_profile,
        )
        row["pre_rerank_score"] = row["final_score"]
        rows.append(row)

    rows.sort(key=lambda item: item["final_score"], reverse=True)
    rows = agent.rerank_results(
        case.question,
        rows,
        query_profile,
        limit=max(case.top_k * 4, agent.RERANK_LIMIT),
    )
    return agent.pack_context_results(rows, case.top_k)


def build_prompt(question, rows, version):
    context = agent.build_context(rows)
    return f"""你是一个可以使用知识库和网页资料的 RAG Agent。

评估版本：{version}
请根据【资料】回答【用户问题】。
如果资料不足，请明确说资料不足，不要编造。
资料使用规则：
1. 上传资料和上传图片属于用户主动提供的信息，可信优先级最高。
2. 网络资料只作为补充；当网络资料和上传资料冲突时，优先采用上传资料。
3. 基础资料只作为兜底，不能压过用户上传资料和当前联网资料。
4. 如果资料来自网页，请提醒用户网页信息可能会变化。

【资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先给结论
2. 再用 2-4 条解释关键依据
3. 最后列出参考来源
"""


def ask_model(prompt):
    if os.getenv("SKIP_MODEL", "0") == "1":
        return "已跳过模型调用；本报告只评估进入 messages 的资料差异。"

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "未设置 DEEPSEEK_API_KEY；本报告只评估进入 messages 的资料差异。"

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=380,
    )
    return response.choices[0].message.content


def compact_sources(rows):
    return [
        {
            "source": row["source"],
            "source_type": row["source_type"],
            "final_score": round(row.get("final_score", 0), 4),
            "pre_rerank_score": round(row.get("pre_rerank_score", 0), 4),
            "freshness_score": round(row.get("freshness_score", 0), 2),
            "answerability_score": round(row.get("answerability_score", 0), 2),
            "query_intent": row.get("query_intent", ""),
            "context_order": row.get("context_order", 0),
            "vector_rank": row.get("vector_rank"),
            "bm25_rank": row.get("bm25_rank"),
            "document": row["document"],
        }
        for row in rows
    ]


def judge(case, rows, answer):
    selected_sources = [row["source"] for row in rows]
    selected_text = "\n".join(row["document"] for row in rows)
    normalized_answer = re.sub(r"\s+", "", answer or "")
    return {
        "expected_source_selected": case.expected_source in selected_sources,
        "expected_text_in_context": case.expected_text in selected_text,
        "expected_text_in_answer": case.expected_text.replace(" ", "") in normalized_answer,
        "selected_sources": selected_sources,
    }


def render_sources_table(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['context_order']))}</td>"
            f"<td>{html.escape(row['source'])}</td>"
            f"<td>{html.escape(row['source_type'])}</td>"
            f"<td>{html.escape(row.get('query_intent', ''))}</td>"
            f"<td>{row.get('final_score', 0):.4f}</td>"
            f"<td>{row.get('freshness_score', 0):.2f}</td>"
            f"<td>{row.get('answerability_score', 0):.2f}</td>"
            f"<td>{html.escape(row['document'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>#</th><th>来源</th><th>类型</th><th>意图</th><th>最终分</th><th>新鲜度</th><th>答案性</th><th>内容</th>"
        "</tr></thead><tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def status_badge(ok, true_text="通过", false_text="需关注"):
    label = true_text if ok else false_text
    class_name = "ok" if ok else "warn"
    return f'<span class="badge {class_name}">{html.escape(label)}</span>'


def source_pills(sources):
    return "".join(f'<span class="pill">{html.escape(source)}</span>' for source in sources)


def top_source(sources):
    if not sources:
        return "无"
    return sources[0]["source"]


def answer_excerpt(answer):
    text = re.sub(r"\s+", " ", answer or "").strip()
    if len(text) <= 260:
        return text
    return text[:260] + "..."


def render_source_compare(a_sources, b_sources):
    max_len = max(len(a_sources), len(b_sources))
    rows = []

    def score_text(row, key, digits):
        if not row:
            return ""
        return f"{row.get(key, 0):.{digits}f}"

    for index in range(max_len):
        a = a_sources[index] if index < len(a_sources) else None
        b = b_sources[index] if index < len(b_sources) else None
        changed = bool(a and b and a["source"] != b["source"])
        row_class = "changed" if changed else ""
        rows.append(
            f"""
            <tr class="{row_class}">
              <td>{index + 1}</td>
              <td>{html.escape(a['source']) if a else ''}</td>
              <td>{score_text(a, 'final_score', 4)}</td>
              <td>{score_text(a, 'answerability_score', 2)}</td>
              <td>{html.escape(b['source']) if b else ''}</td>
              <td>{score_text(b, 'final_score', 4)}</td>
              <td>{score_text(b, 'answerability_score', 2)}</td>
            </tr>
            """
        )

    return (
        "<table class=\"compare-table\"><thead><tr>"
        "<th>顺序</th><th>A 来源</th><th>A 分</th><th>A 答案性</th>"
        "<th>B 来源</th><th>B 分</th><th>B 答案性</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_report(results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows_html = []
    b_context_pass = sum(1 for item in results if item["B"]["judge"]["expected_text_in_context"])
    b_answer_pass = sum(1 for item in results if item["B"]["judge"]["expected_text_in_answer"])
    changed_top = sum(1 for item in results if top_source(item["A"]["sources"]) != top_source(item["B"]["sources"]))

    for item in results:
        a_judge = item["A"]["judge"]
        b_judge = item["B"]["judge"]
        a_top = top_source(item["A"]["sources"])
        b_top = top_source(item["B"]["sources"])
        top_changed = a_top != b_top
        rows_html.append(
            f"""
            <section class="case-card">
              <div class="case-head">
                <div>
                  <h2>{html.escape(item['case'])}</h2>
                  <p class="question">{html.escape(item['question'])}</p>
                </div>
                <div class="badges">
                  {status_badge(b_judge['expected_text_in_context'], 'B 上下文命中', 'B 上下文未命中')}
                  {status_badge(b_judge['expected_text_in_answer'], 'B 回答命中', 'B 回答未命中')}
                  {status_badge(top_changed, '首位有变化', '首位无变化')}
                </div>
              </div>

              <div class="summary-grid">
                <div class="summary-box">
                  <div class="label">A 首位资料</div>
                  <div class="value">{html.escape(a_top)}</div>
                </div>
                <div class="summary-box highlight">
                  <div class="label">B 首位资料</div>
                  <div class="value">{html.escape(b_top)}</div>
                </div>
                <div class="summary-box">
                  <div class="label">A 命中情况</div>
                  <div>{status_badge(a_judge['expected_text_in_context'], '上下文命中', '上下文未命中')} {status_badge(a_judge['expected_text_in_answer'], '回答命中', '回答未命中')}</div>
                </div>
                <div class="summary-box highlight">
                  <div class="label">B 命中情况</div>
                  <div>{status_badge(b_judge['expected_text_in_context'], '上下文命中', '上下文未命中')} {status_badge(b_judge['expected_text_in_answer'], '回答命中', '回答未命中')}</div>
                </div>
              </div>

              <h3>资料顺序对比</h3>
              {render_source_compare(item['A']['sources'], item['B']['sources'])}

              <details>
                <summary>查看 A/B 完整资料内容</summary>
                <div class="grid">
                  <div>
                    <h3>A 优化前：固定 RRF + 来源配额</h3>
                    {render_sources_table(item['A']['sources'])}
                  </div>
                  <div>
                    <h3>B 优化后：意图权重 + Reranker + 去重 + Context Packing</h3>
                    {render_sources_table(item['B']['sources'])}
                  </div>
                </div>
              </details>

              <h3>回答对比</h3>
              <div class="answer-grid">
                <div>
                  <h4>A 回答摘要</h4>
                  <p>{html.escape(answer_excerpt(item['A']['answer']))}</p>
                </div>
                <div>
                  <h4>B 回答摘要</h4>
                  <p>{html.escape(answer_excerpt(item['B']['answer']))}</p>
                </div>
              </div>
              <details>
                <summary>查看完整回答</summary>
                <div class="grid">
                  <pre>{html.escape(item['A']['answer'])}</pre>
                  <pre>{html.escape(item['B']['answer'])}</pre>
                </div>
              </details>
            </section>
            """
        )

    html_text = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>RAG Pipeline A/B 评估报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f4f6f8; color: #1f2933; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin: 0; }}
    h3 {{ margin-top: 22px; }}
    .hero {{ background: #ffffff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 24px; margin-bottom: 18px; }}
    .metric-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }}
    .metric {{ background: #f8fafc; border: 1px solid #e4e7eb; border-radius: 8px; padding: 14px; }}
    .metric .num {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .case-card {{ background: #ffffff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 22px; margin: 18px 0; }}
    .case-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; }}
    .question {{ margin: 8px 0 0; color: #52606d; }}
    .badges {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .badge.ok {{ background: #e3fcef; color: #176f3d; }}
    .badge.warn {{ background: #fff4e5; color: #8a4b00; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .summary-box {{ border: 1px solid #e4e7eb; border-radius: 8px; padding: 12px; background: #fbfcfd; }}
    .summary-box.highlight {{ border-color: #b7d8ff; background: #f0f7ff; }}
    .label {{ font-size: 12px; color: #66788a; margin-bottom: 6px; }}
    .value {{ font-weight: 650; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }}
    .answer-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f7fa; text-align: left; }}
    .compare-table tr.changed td {{ background: #fffdf0; }}
    pre {{ background: #f6f8fa; padding: 12px; white-space: pre-wrap; border-radius: 6px; }}
    details {{ margin-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 650; color: #334e68; }}
    @media (max-width: 1000px) {{ .grid, .summary-grid, .answer-grid, .metric-row {{ grid-template-columns: 1fr; }} .case-head {{ flex-direction: column; }} .badges {{ justify-content: flex-start; }} }}
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <h1>RAG Pipeline A/B 评估报告</h1>
      <p>A 是优化前固定检索策略；B 是优化后的意图识别、动态权重、Reranker、时间新鲜度、去重和 Context Packing。</p>
      <div class="metric-row">
        <div class="metric"><div class="label">评估 Case</div><div class="num">{len(results)}</div></div>
        <div class="metric"><div class="label">B 上下文命中</div><div class="num">{b_context_pass}/{len(results)}</div></div>
        <div class="metric"><div class="label">首位资料变化</div><div class="num">{changed_top}/{len(results)}</div></div>
      </div>
    </div>
    {''.join(rows_html)}
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(html_text, encoding="utf-8")


def main():
    results = []

    for case in CASES:
        a_rows = old_pipeline(case)
        b_rows = new_pipeline(case)
        a_prompt = build_prompt(case.question, a_rows, "A")
        b_prompt = build_prompt(case.question, b_rows, "B")
        a_answer = ask_model(a_prompt)
        b_answer = ask_model(b_prompt)

        results.append({
            "case": case.name,
            "question": case.question,
            "A": {
                "sources": compact_sources(a_rows),
                "answer": a_answer,
                "judge": judge(case, a_rows, a_answer),
            },
            "B": {
                "sources": compact_sources(b_rows),
                "answer": b_answer,
                "judge": judge(case, b_rows, b_answer),
            },
        })

    render_report(results)
    print(json.dumps({
        "report": str(REPORT_PATH),
        "cases": len(results),
        "skip_model": os.getenv("SKIP_MODEL", "0") == "1" or not os.getenv("DEEPSEEK_API_KEY"),
        "summary": [
            {
                "case": item["case"],
                "A_sources": item["A"]["judge"]["selected_sources"],
                "B_sources": item["B"]["judge"]["selected_sources"],
                "A_expected_source_selected": item["A"]["judge"]["expected_source_selected"],
                "B_expected_source_selected": item["B"]["judge"]["expected_source_selected"],
                "A_expected_text_in_answer": item["A"]["judge"]["expected_text_in_answer"],
                "B_expected_text_in_answer": item["B"]["judge"]["expected_text_in_answer"],
            }
            for item in results
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
