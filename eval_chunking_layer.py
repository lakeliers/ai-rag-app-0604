import html
import json
import os
from pathlib import Path

from openai import OpenAI

import chunking_layer
from parsing_layer import ParsedSection


REPORT_PATH = Path("reports/chunking_layer_report.html")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def make_cases():
    long_text = ParsedSection(
        section_title="价格策略",
        content_type="docx_section",
        text="""会员首月价格为 129 元，次月恢复原价。

基础版首月 99 元，适合轻量用户。专业版首月 129 元，包含高级分析能力。企业版首月 399 元，包含团队管理和专属支持。

退款规则：用户在购买后 7 天内未使用核心权益，可以申请退款。年付用户享受 8 折优惠。""",
    )

    table_text = ParsedSection(
        section_title="定价方案 行 1-6",
        content_type="table",
        sheet="定价方案",
        row_start=1,
        row_end=6,
        text="""套餐 | 首月价格 | 原价
基础版 | 99 | 199
专业版 | 129 | 299
企业版 | 399 | 599
教育版 | 59 | 129
团队版 | 499 | 899""",
    )

    summary_text = ParsedSection(
        section_title="会员商业化策略",
        content_type="docx_section",
        text="""本节说明会员商业化策略。基础版首月价格为 99 元，专业版首月价格为 129 元，企业版首月价格为 399 元。
次月恢复原价，年付用户享受 8 折优惠。产品预计 6 月 10 日上线，支付页预计 6 月 15 日上线。
客服升级后首响时间控制在 30 秒以内，优先处理会员用户咨询。退款规则为购买后 7 天内未使用核心权益可申请退款。""",
    )

    return [
        {
            "name": "recursive_text",
            "title": "Recursive 文本切分",
            "section": long_text,
            "why": "普通文本优先按段落和句子切，避免固定长度把句子切断。",
        },
        {
            "name": "table_aware",
            "title": "Table-aware 表格切分",
            "section": table_text,
            "why": "表格 chunk 必须重复表头，否则数字会和列名断开。",
        },
        {
            "name": "parent_summary",
            "title": "Parent-child + 摘要 chunk",
            "section": summary_text,
            "why": "小块负责检索，parent_text 负责回答；摘要 chunk 用更短文本增强召回。",
        },
    ]


def fixed_chunks(section, chunk_size=90):
    text = chunking_layer.format_section_text(section)
    return [
        {
            "text": chunk,
            "chunk_type": "fixed",
            "parent_id": "",
            "parent_text": "",
        }
        for chunk in chunking_layer.split_text_fixed(text, chunk_size=chunk_size, chunk_overlap=20)
    ]


def advanced_chunks(section, source, section_index, chunk_size=140):
    return [
        {
            "text": chunk.text,
            "chunk_type": chunk.chunk_type,
            "parent_id": chunk.parent_id,
            "parent_text": chunk.parent_text,
        }
        for chunk in chunking_layer.chunk_section(
            section,
            source=source,
            section_index=section_index,
            chunk_size=chunk_size,
            chunk_overlap=20,
        )
    ]


def summarize_for_report(section):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "未设置 DEEPSEEK_API_KEY，本次报告跳过真实摘要生成。"

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"""请把下面资料压缩成适合知识库检索的摘要。
要求保留关键数字、时间、规则，控制在 120 字以内，不要加入原文没有的信息。

资料：
{section.text}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=160,
    )
    return response.choices[0].message.content.strip()


def render_chunk_cards(chunks):
    cards = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = []
        if chunk.get("chunk_type"):
            metadata.append(f"chunk_type={chunk['chunk_type']}")
        if chunk.get("parent_id"):
            metadata.append(f"parent_id={chunk['parent_id']}")
        cards.append(
            f"""
            <div class="chunk-card">
              <div class="chunk-head">Chunk {index}</div>
              <div class="meta">{html.escape(' | '.join(metadata))}</div>
              <pre>{html.escape(chunk['text'])}</pre>
            </div>
            """
        )
    return "\n".join(cards)


def render_report(results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections_html = []
    for item in results:
        sections_html.append(
            f"""
            <section>
              <h2>{html.escape(item['title'])}</h2>
              <p>{html.escape(item['why'])}</p>
              <div class="input-box">
                <div class="label">输入 ParsedSection</div>
                <pre>{html.escape(item['input'])}</pre>
              </div>
              <div class="grid">
                <div>
                  <h3>A：固定长度切分</h3>
                  {render_chunk_cards(item['fixed'])}
                </div>
                <div>
                  <h3>B：高级切分</h3>
                  {render_chunk_cards(item['advanced'])}
                </div>
              </div>
              <div class="summary-box">
                <div class="label">摘要 chunk 示例</div>
                <pre>{html.escape(item['summary'])}</pre>
              </div>
            </section>
            """
        )

    html_text = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>RAG 切分层评估报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f4f6f8; color: #1f2933; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    .hero, section {{ background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 22px; margin-bottom: 18px; }}
    h1, h2 {{ margin: 0 0 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .input-box, .summary-box {{ border: 1px solid #dfe3e8; border-radius: 8px; background: #fbfcfd; padding: 12px; margin: 14px 0; }}
    .label {{ font-size: 12px; color: #66788a; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }}
    .chunk-card {{ border: 1px solid #dfe3e8; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
    .chunk-head {{ background: #f5f7fa; padding: 8px 10px; font-weight: 700; }}
    .meta {{ color: #66788a; font-size: 12px; padding: 8px 10px 0; }}
    pre {{ margin: 0; padding: 10px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.5; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <h1>RAG 切分层评估报告</h1>
      <p>本报告对比固定长度切分与高级切分：recursive、table-aware、parent-child 和摘要 chunk。</p>
    </div>
    {''.join(sections_html)}
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(html_text, encoding="utf-8")


def main():
    results = []
    for index, case in enumerate(make_cases()):
        section = case["section"]
        summary = summarize_for_report(section) if case["name"] == "parent_summary" else "该案例不需要摘要生成。"
        advanced = advanced_chunks(section, source="chunk_eval", section_index=index)
        if case["name"] == "parent_summary" and summary and not summary.startswith("未设置"):
            advanced.insert(0, {
                "text": f"摘要：{summary}",
                "chunk_type": "summary",
                "parent_id": advanced[0]["parent_id"] if advanced else "",
                "parent_text": section.text,
            })
        results.append({
            "title": case["title"],
            "why": case["why"],
            "input": json.dumps({
                "section_title": section.section_title,
                "content_type": section.content_type,
                "sheet": section.sheet,
                "row_start": section.row_start,
                "row_end": section.row_end,
                "text": section.text,
            }, ensure_ascii=False, indent=2),
            "fixed": fixed_chunks(section),
            "advanced": advanced,
            "summary": summary,
        })

    render_report(results)
    print(json.dumps({
        "report": str(REPORT_PATH),
        "cases": len(results),
        "summary_enabled": bool(os.getenv("DEEPSEEK_API_KEY")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
