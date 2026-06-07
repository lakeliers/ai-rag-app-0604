import json
import os
from dataclasses import dataclass

from openai import OpenAI


MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


@dataclass
class Source:
    source: str
    source_type: str
    content: str


@dataclass
class Case:
    name: str
    question: str
    sources: list[Source]
    expected_text: str
    forbidden_text: str
    expected_source_type: str


CASES = [
    Case(
        name="conflict_launch_date",
        question="产品什么时候上线？",
        sources=[
            Source("网页：外部新闻", "web", "该产品预计在 6 月 20 日上线，面向公开市场发布。"),
            Source("上传：PRD.md", "upload", "内部 PRD 明确写着：产品上线时间为 6 月 10 日。"),
        ],
        expected_text="6 月 10",
        forbidden_text="6 月 20",
        expected_source_type="upload",
    ),
    Case(
        name="conflict_price",
        question="会员首月价格是多少？",
        sources=[
            Source("网页：促销报道", "web", "公开报道称会员首月价格为 199 元。"),
            Source("上传：定价方案.xlsx", "upload", "最新定价方案：会员首月价格为 129 元，次月恢复原价。"),
        ],
        expected_text="129",
        forbidden_text="199",
        expected_source_type="upload",
    ),
    Case(
        name="upload_has_answer",
        question="客服升级后的目标响应时间是多少？",
        sources=[
            Source("网页：行业文章", "web", "行业客服响应时间通常在 5 分钟以内。"),
            Source("上传：客服改版需求.docx", "upload", "客服升级目标：首响时间控制在 30 秒以内。"),
        ],
        expected_text="30 秒",
        forbidden_text="5 分钟",
        expected_source_type="upload",
    ),
    Case(
        name="web_fallback",
        question="AI Agent 市场规模在 2025 年大约是多少？",
        sources=[
            Source("网页：市场报告", "web", "报告显示，2025 年全球 AI Agent 市场规模约为 73.8 亿美元。"),
            Source("基础资料：学习笔记", "local", "RAG 是检索增强生成，Chroma 是向量数据库。"),
        ],
        expected_text="73.8",
        forbidden_text="资料不足",
        expected_source_type="web",
    ),
]


def source_label(source):
    if source.source_type == "upload" and source.source.startswith("图片："):
        return "上传图片｜优先"
    if source.source_type == "upload":
        return "上传资料｜优先"
    if source.source_type == "web":
        return "网络资料｜补充"
    if source.source_type == "local":
        return "基础资料｜兜底"
    return "其他资料｜参考"


def build_context(case, version):
    parts = []
    for index, source in enumerate(case.sources, start=1):
        priority_line = ""
        if version == "B":
            priority_line = f"优先级：{source_label(source)}\n"

        parts.append(
            f"""资料 {index}：
来源：{source.source}
类型：{source.source_type}
{priority_line}内容：{source.content}
"""
        )
    return "\n---\n".join(parts)


def build_prompt(case, version):
    context = build_context(case, version)

    if version == "A":
        rules = """请根据【资料】回答【用户问题】。
如果资料不足，请明确说资料不足，不要编造。
如果资料来自网页，请提醒用户网页信息可能会变化。"""
    else:
        rules = """请根据【资料】回答【用户问题】。
如果资料不足，请明确说资料不足，不要编造。
资料使用规则：
1. 上传资料和上传图片属于用户主动提供的信息，可信优先级最高。
2. 网络资料只作为补充；当网络资料和上传资料冲突时，优先采用上传资料。
3. 基础资料只作为兜底，不能压过用户上传资料和当前联网资料。
4. 如果资料来自网页，请提醒用户网页信息可能会变化。"""

    return f"""你是一个可以使用知识库和网页资料的 RAG Agent。

{rules}

【资料】
{context}

【用户问题】
{case.question}

【回答要求】
1. 先给结论
2. 再用 2-4 条解释关键依据
3. 最后列出参考来源
"""


def ask_model(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=350,
    )
    return response.choices[0].message.content


def judge(case, answer):
    normalized_answer = re_space(answer)
    expected_text = re_space(case.expected_text)
    forbidden_text = re_space(case.forbidden_text)
    conclusion_text = re_space(extract_conclusion(answer))

    has_expected = expected_text in normalized_answer
    mentions_forbidden = forbidden_text in normalized_answer if forbidden_text else False
    adopts_forbidden = forbidden_text in conclusion_text if forbidden_text else False
    mentions_upload = "上传" in answer or "PRD" in answer or "定价方案" in answer or "客服改版需求" in answer
    mentions_web = "网页" in answer or "报告" in answer or "市场报告" in answer

    if case.expected_source_type == "upload":
        source_ok = mentions_upload
    elif case.expected_source_type == "web":
        source_ok = mentions_web
    else:
        source_ok = True

    return {
        "has_expected_text": has_expected,
        "mentions_forbidden_text": mentions_forbidden,
        "adopts_forbidden_text": adopts_forbidden,
        "source_ok": source_ok,
        "pass": has_expected and not adopts_forbidden and source_ok,
    }


def re_space(text):
    return "".join((text or "").split())


def extract_conclusion(answer):
    separators = [
        "\n\n",
        "**关键依据",
        "关键依据",
        "### 关键依据",
        "参考来源",
    ]

    first_part = answer
    for separator in separators:
        if separator in first_part:
            first_part = first_part.split(separator, 1)[0]

    return first_part[:220]


def summarize(rows):
    total = len(rows)
    passed = sum(1 for row in rows if row["judge"]["pass"])
    expected = sum(1 for row in rows if row["judge"]["has_expected_text"])
    forbidden = sum(1 for row in rows if row["judge"]["mentions_forbidden_text"])
    adopts_forbidden = sum(1 for row in rows if row["judge"]["adopts_forbidden_text"])
    source_ok = sum(1 for row in rows if row["judge"]["source_ok"])
    return {
        "total": total,
        "pass_rate": passed / total,
        "expected_text_rate": expected / total,
        "mentions_forbidden_text_rate": forbidden / total,
        "adopts_forbidden_text_rate": adopts_forbidden / total,
        "source_ok_rate": source_ok / total,
    }


def mechanism_eval():
    a_prompt = build_prompt(CASES[0], "A")
    b_prompt = build_prompt(CASES[0], "B")
    return {
        "A_has_priority_rules": "上传资料和上传图片" in a_prompt,
        "B_has_priority_rules": "上传资料和上传图片" in b_prompt,
        "A_has_priority_labels": "优先级：" in a_prompt,
        "B_has_priority_labels": "优先级：" in b_prompt,
    }


def main():
    all_results = {}
    for version in ["A", "B"]:
        rows = []
        for case in CASES:
            prompt = build_prompt(case, version)
            answer = ask_model(prompt)
            rows.append({
                "case": case.name,
                "answer": answer,
                "judge": judge(case, answer),
            })
        all_results[version] = {
            "summary": summarize(rows),
            "rows": rows,
        }

    report = {
        "definition": {
            "A": "优化前：无资料优先级规则、无优先级标签",
            "B": "优化后：有上传优先/网络补充/基础兜底规则和标签",
        },
        "mechanism_eval": mechanism_eval(),
        "model_eval": all_results,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
