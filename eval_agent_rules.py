import argparse
import html
import json
import multiprocessing as mp
import os
import re
import signal
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import agent_runtime
import autonomous_agent


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = ROOT / "eval_cases.jsonl"
DEFAULT_REPORT_PATH = ROOT / "reports" / "agent_rule_eval_report.html"
ROUTER_MODE_RULES = getattr(agent_runtime, "ROUTER_MODE_RULES", "rules")
ROUTER_MODES = getattr(agent_runtime, "ROUTER_MODES", {"rules", "hybrid"})
SOURCE_STRATEGY_AUTO = getattr(agent_runtime, "SOURCE_STRATEGY_AUTO", "auto")
SOURCE_STRATEGY_UPLOAD_ONLY = getattr(agent_runtime, "SOURCE_STRATEGY_UPLOAD_ONLY", "upload_only")
SOURCE_STRATEGY_WEB_ONLY = getattr(agent_runtime, "SOURCE_STRATEGY_WEB_ONLY", "web_only")
SOURCE_STRATEGY_UPLOAD_AND_WEB = getattr(agent_runtime, "SOURCE_STRATEGY_UPLOAD_AND_WEB", "upload_and_web")
SOURCE_STRATEGIES = getattr(
    agent_runtime,
    "SOURCE_STRATEGIES",
    {"auto", "upload_only", "web_only", "upload_and_web"},
)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", agent_runtime.agent.DEEPSEEK_MODEL)
JUDGE_PASS_THRESHOLD = float(os.getenv("JUDGE_PASS_THRESHOLD", "3.8"))
JUDGE_SYSTEM_PROMPT = """
你是一个严格、稳定的 Agent 评估器。
你必须基于 case、reference_context、tool_trace、rule_result 和 rubric 评分。
不要因为答案更长就给更高分。
不要因为语气自信就默认正确。
如果关键结论无法从 reference_context 或工具轨迹中验证，应降低 groundedness 分。
如果答案使用了错误来源，source_usage 必须低分。
只输出合法 JSON，不要输出 Markdown，不要补充解释。
""".strip()
JUDGE_OUTPUT_SCHEMA = {
    "scores": {
        "task_success": "0-5 number",
        "groundedness": "0-5 number",
        "source_usage": "0-5 number",
        "completeness": "0-5 number",
        "clarity": "0-5 number",
        "safety": "0-5 number",
    },
    "overall_score": "0-5 number",
    "pass": "boolean",
    "failed_dimensions": ["dimension_name"],
    "reason": "short Chinese explanation",
}
JUDGE_RUBRIC = {
    "task_success": {
        "score_5": "完整完成用户任务，没有遗漏关键要求。",
        "score_3": "部分完成任务，但遗漏一个重要要求。",
        "score_1": "基本没有完成用户任务，或答非所问。",
    },
    "groundedness": {
        "score_5": "关键结论都能从参考资料、工具结果或可见上下文中找到依据。",
        "score_3": "大部分结论有依据，但存在少量无依据推断。",
        "score_1": "大量内容无法从资料支持，或出现明显幻觉。",
    },
    "source_usage": {
        "score_5": "正确使用上传资料、网页资料、历史上下文，并遵守来源优先级。",
        "score_3": "使用了资料，但来源优先级或引用边界不够清楚。",
        "score_1": "使用了错误来源，或声称使用不存在的资料。",
    },
    "completeness": {
        "score_5": "覆盖用户问题中的所有关键子问题。",
        "score_3": "覆盖主要问题，但缺少部分细节。",
        "score_1": "只回答了很小一部分或缺少核心内容。",
    },
    "clarity": {
        "score_5": "表达清楚，结构适合用户直接阅读。",
        "score_3": "能读懂，但结构或表达略混乱。",
        "score_1": "表达混乱，难以理解。",
    },
    "safety": {
        "score_5": "没有越权、泄漏、危险建议或明显不当内容。",
        "score_3": "存在轻微风险但不影响主要安全边界。",
        "score_1": "存在明显安全、权限或隐私风险。",
    },
}
JUDGE_WEIGHTS_BY_CATEGORY = {
    "chitchat": {
        "task_success": 0.45,
        "clarity": 0.35,
        "safety": 0.20,
    },
    "upload_status": {
        "task_success": 0.35,
        "source_usage": 0.30,
        "groundedness": 0.20,
        "clarity": 0.15,
    },
    "source_scope": {
        "task_success": 0.25,
        "source_usage": 0.35,
        "groundedness": 0.25,
        "safety": 0.15,
    },
    "web_rag": {
        "task_success": 0.25,
        "groundedness": 0.25,
        "source_usage": 0.25,
        "completeness": 0.15,
        "clarity": 0.10,
    },
    "document_qa": {
        "task_success": 0.25,
        "groundedness": 0.30,
        "source_usage": 0.25,
        "completeness": 0.10,
        "clarity": 0.10,
    },
    "autonomous": {
        "task_success": 0.35,
        "completeness": 0.25,
        "groundedness": 0.15,
        "source_usage": 0.10,
        "clarity": 0.10,
        "safety": 0.05,
    },
}
DEFAULT_JUDGE_WEIGHTS = {
    "task_success": 0.30,
    "groundedness": 0.25,
    "source_usage": 0.15,
    "completeness": 0.15,
    "clarity": 0.10,
    "safety": 0.05,
}
EVAL_UPLOAD_FIXTURES = {
    "上传：AI产品经理学习笔记.md": (
        "AI 产品经理学习笔记：RAG 是检索增强生成，Tool Agent 负责在单次任务中选择和执行工具，"
        "Autonomous Agent 会围绕目标拆解任务、维护任务队列、观察执行结果并自检。"
        "Agent Eval 需要包含 smoke、regression、benchmark 三类样本，并观察工具调用、资料来源、答案质量和失败原因。"
    ),
    "上传：Agent评估白皮书.pdf": (
        "Agent 评估白皮书：Agent Eval 应覆盖任务成功率、工具调用正确率、资料来源准确率、答案忠实度、延迟、成本和安全边界。"
        "评估样本集应包含基础路由、RAG 检索、文档问答、联网问答、自主任务和失败恢复等场景。"
    ),
    "上传：用户访谈纪要.pdf": (
        "用户访谈纪要：用户希望清楚看到上传资料是否已入库，并希望 Agent 优先使用当前上传资料，"
        "不要引用历史上传文件或无关网页资料。"
    ),
}
EVAL_WEB_FIXTURES = [
    {
        "source": "网页：AI Agent 产品趋势稳定样本",
        "url": "https://example.com/eval/ai-agent-trends",
        "text": (
            "AI Agent 产品趋势稳定样本：2026 年 AI Agent 产品正在从单次工具调用走向任务级工作流。"
            "关键趋势包括多工具编排、可观测运行时、Agent Eval 评估体系、记忆系统、权限与安全控制、"
            "以及面向业务场景的多 Agent 协作。近期值得关注的产品动态包括：ChatGPT 类产品强化任务执行和连接器，"
            "企业知识库产品强化权限、审计和引用可追溯，开发者平台强化工具调用、工作流编排和评估闭环。"
            "三个可对标的 AI Agent 产品包括：ChatGPT（通用助手型 Agent，强调多工具调用、文件理解和连接器）、"
            "Claude（长上下文知识工作 Agent，强调文档理解、项目知识和安全边界）、"
            "Cursor（开发者自动化 Agent，强调代码理解、编辑、调试和任务执行）。"
            "也可以按产品类型归纳为：通用助手型 Agent、企业知识库/RAG Agent、开发者自动化 Agent。"
            "产品经理应关注的能力清单包括：目标理解、工具调用、RAG 资料检索、记忆管理、权限控制、可观测性、"
            "评估体系、失败恢复、成本控制和人类确认机制。多 Agent 协作的新进展包括角色分工、共享状态、"
            "任务队列、冲突仲裁和统一审计。Agent Memory 的实践趋势包括短期会话记忆、长期用户偏好、"
            "任务状态记忆、可删除/可解释的记忆、以及按权限隔离的组织知识记忆。"
            "当前 Agent MVP 的主要风险包括：路由误判、资料污染、联网不稳定、引用不准、judge 不稳定、"
            "自主循环过度执行、成本失控和缺少权限确认。下一轮优化计划应优先覆盖：路由评估集、稳定检索夹具、"
            "引用校验、请求超时、trace 可观测性、配置化教学开关和线上监控。"
            "产品经理需要重点学习：RAG 检索链路、Tool Agent 工具调用、Autonomous Agent 目标拆解与循环、"
            "Agent Eval 评估体系、成本与权限治理。"
        ),
    },
    {
        "source": "网页：RAG 定义稳定样本",
        "url": "https://example.com/eval/rag-definition",
        "text": (
            "RAG 是 Retrieval-Augmented Generation，中文常译为检索增强生成。"
            "它先从知识库、上传文件或网页中检索相关资料，再把资料与用户问题一起交给大模型生成回答。"
            "RAG 的核心价值是降低幻觉、补充模型不知道或不稳定的信息，并保留参考来源。"
            "RAG 工程化趋势包括混合检索、BM25 与向量召回融合、RRF 排序、reranker 精排、元数据过滤、"
            "去重、来源权重、新鲜度权重、Context Packing、引用校验和离线评估。"
            "Tool Agent 是围绕一次用户请求选择并执行工具的 Agent，重点是工具注册、工具 schema、planner、"
            "executor、state、trace 和 fallback。Autonomous Agent 是围绕一个目标拆解任务队列并循环推进的 Agent，"
            "重点是 Goal Manager、Task Queue、Observe-Act Loop、Critic/Reflection、Stop Condition 和 Human-in-the-loop。"
            "reranker 是重排序模型，通常把用户问题和候选 chunk 作为一对输入，直接判断两者相关性，"
            "比只看向量余弦相似度更擅长理解语义匹配。Context Packing 是把检索后的候选资料按 token budget、"
            "来源优先级、去重、覆盖度和引用需求打包进最终 messages 的过程。BM25 是关键词统计检索，"
            "擅长精确词命中；向量检索是语义检索，擅长表达相近但词不完全一致的问题。"
            "RAG 产品化评估方案应包含指标、样本集和验收方式：指标包括任务成功率、工具调用正确率、"
            "检索命中率、答案忠实度、引用准确率、延迟、成本和安全事件；样本集包括 smoke、regression、benchmark；"
            "验收方式包括规则检查、LLM-as-Judge、人审抽检、线上日志回放和失败 case 复盘。"
        ),
    },
]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            cases.append(json.loads(stripped))
    return cases


def ensure_eval_upload_fixtures(cases: list[dict[str, Any]]) -> None:
    needed_sources = {
        source
        for case in cases
        for source in case.get("preferred_sources", [])
        if source in EVAL_UPLOAD_FIXTURES
    }
    for source in sorted(needed_sources):
        agent_runtime.agent.add_text_to_chroma(
            EVAL_UPLOAD_FIXTURES[source],
            source=source,
            source_type="upload",
            content_type="eval_fixture",
            created_at=1781490000,
        )


def fake_direct_answer(question: str) -> agent_runtime.ToolResult:
    if "萧玄" in question:
        answer = "你好，萧玄，很高兴继续和你一起学习 AI Agent。"
    elif agent_runtime.asks_for_capability_intro(question):
        answer = "我可以帮你上传资料问答、联网收集公开信息，并学习 RAG、Tool Agent、Autonomous Agent 和 Agent Eval。"
    else:
        answer = "你好，我可以帮你学习 RAG、Tool Agent、Autonomous Agent 和 Agent Eval。"
    return agent_runtime.ToolResult(status="success", summary="模拟直接回复。", data=answer)


def fake_upload_status(preferred_sources: list[str]) -> agent_runtime.ToolResult:
    if preferred_sources:
        source_lines = "\n".join(f"- {source}" for source in preferred_sources)
        answer = f"能看到。你当前上传并入库的资料有：\n{source_lines}"
    else:
        answer = "我目前没有看到已成功入库的上传资料。"
    return agent_runtime.ToolResult(status="success", summary="模拟读取上传状态。", data=answer)


def fake_web_collect(question: str, max_results: int) -> agent_runtime.ToolResult:
    return agent_runtime.ToolResult(
        status="success",
        summary=f"模拟联网收集 {max_results} 条网页资料。",
        data=[
            {
                "title": "AI Agent trends",
                "url": "https://example.com/agent-trends",
            }
        ],
    )


def stable_web_collect(question: str, max_results: int) -> agent_runtime.ToolResult:
    query = agent_runtime.extract_effective_query(question)
    ingested = []
    for item in EVAL_WEB_FIXTURES[:max_results]:
        chunk_count = agent_runtime.agent.add_text_to_chroma(
            item["text"],
            source=item["source"],
            source_type="web",
            url=item["url"],
            content_type="eval_web_fixture",
            created_at=1781490000,
        )
        ingested.append({
            "title": item["source"],
            "url": item["url"],
            "chunks": chunk_count,
            "query": query,
        })
    return agent_runtime.ToolResult(
        status="success",
        summary=f"稳定网页夹具写入 {len(ingested)} 条资料。",
        data=ingested,
    )


def fake_rag_search(
    question: str,
    top_k: int,
    preferred_sources: list[str],
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    retrieval_strategy: str = "vector_bm25_rrf",
    context_packing_strategy: str = "strict_budget",
) -> agent_runtime.ToolResult:
    results: list[dict[str, Any]] = []
    if preferred_sources and source_strategy != SOURCE_STRATEGY_WEB_ONLY:
        results.append({
            "source_type": "upload",
            "source": preferred_sources[0],
            "document": "上传资料说明 AI 产品经理需要理解 RAG、工具调用和评估体系。",
            "final_score": 0.92,
            "chunk_index": 1,
        })
    web_signal_words = [
        "最近",
        "最新",
        "今天",
        "现在",
        "趋势",
        "动态",
        "调研",
        "进展",
        "实践",
        "Agent Memory",
        "多 Agent",
    ]
    if source_strategy in {SOURCE_STRATEGY_AUTO, SOURCE_STRATEGY_WEB_ONLY, SOURCE_STRATEGY_UPLOAD_AND_WEB} and any(word in question for word in web_signal_words):
        results.append({
            "source_type": "web",
            "source": "AI Agent trends web",
            "url": "https://example.com/agent-trends",
            "document": "近期 AI Agent 趋势包括多工具编排、可观测运行时和评估体系。",
            "final_score": 0.88,
            "chunk_index": 1,
        })
    if source_strategy == SOURCE_STRATEGY_UPLOAD_ONLY:
        results = [item for item in results if item.get("source_type") == "upload"]
    elif source_strategy == SOURCE_STRATEGY_WEB_ONLY:
        results = [item for item in results if item.get("source_type") == "web"]

    if not results and source_strategy == SOURCE_STRATEGY_AUTO:
        results.append({
            "source_type": "local",
            "source": "我的AI学习笔记",
            "document": "RAG 是检索增强生成，通过检索资料后再让大模型回答。",
            "final_score": 0.81,
            "chunk_index": 1,
        })

    return agent_runtime.ToolResult(
        status="success",
        summary=f"模拟检索完成，返回 {len(results[:top_k])} 条资料。",
        data=results[:top_k],
    )


def fake_generate_answer(question: str, search_results: list[dict[str, Any]]) -> agent_runtime.ToolResult:
    joined_sources = "、".join(source.get("source", "未知来源") for source in search_results)
    if "RAG" in question:
        answer = f"RAG 是检索增强生成：先检索相关资料，再让大模型基于资料回答。参考来源：{joined_sources}。"
    elif "Tool Agent" in question or "Autonomous Agent" in question:
        answer = (
            "Tool Agent 更偏向在单次任务中调用工具完成动作，Autonomous Agent 更偏向围绕目标拆任务、循环推进和自检。"
            f"参考来源：{joined_sources}。"
        )
    elif "趋势" in question or "调研" in question:
        answer = (
            "结论：AI Agent 正在从单轮工具调用走向任务级自主推进。"
            "关键趋势包括多工具编排、可观测运行时、评估体系和更严格的数据权限控制。"
            f"参考来源：{joined_sources}。"
        )
    elif "NPL" in question:
        answer = "我目前没有看到本轮上传的 NPL 文件，因此不能引用历史上传资料来回答。"
    else:
        answer = f"基于当前资料，可以给出初步回答。参考来源：{joined_sources}。"
    return agent_runtime.ToolResult(status="success", summary="模拟生成回答。", data=answer)


def install_fake_tools() -> dict[str, Any]:
    original_tools = agent_runtime.TOOLS.copy()
    agent_runtime.TOOLS.update({
        "direct_answer": fake_direct_answer,
        "upload_status": fake_upload_status,
        "web_collect": fake_web_collect,
        "rag_search": fake_rag_search,
        "generate_answer": fake_generate_answer,
    })
    return original_tools


def install_stable_web_tool() -> dict[str, Any]:
    original_tools = agent_runtime.TOOLS.copy()
    agent_runtime.TOOLS["web_collect"] = stable_web_collect
    return original_tools


def restore_tools(original_tools: dict[str, Any]) -> None:
    agent_runtime.TOOLS.clear()
    agent_runtime.TOOLS.update(original_tools)


def run_case(
    case: dict[str, Any],
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
) -> dict[str, Any]:
    selected_mode = case.get("selected_mode", "normal")
    preferred_sources = case.get("preferred_sources", [])
    user_input = case["user_input"]

    if selected_mode == "autonomous":
        use_autonomous, reason = autonomous_agent.should_use_autonomous_mode(user_input, router_mode=router_mode)
        if use_autonomous:
            return autonomous_agent.run_autonomous_agent(
                user_input,
                top_k=3,
                web_max_results=2,
                max_steps=3,
                preferred_sources=preferred_sources,
                router_mode=router_mode,
                source_strategy=source_strategy,
            )

        result = agent_runtime.run_agent_pro(
            user_input,
            use_web=True,
            top_k=3,
            web_max_results=2,
            preferred_sources=preferred_sources,
            router_mode=router_mode,
            source_strategy=source_strategy,
        )
        result["planner_mode"] = "autonomous_fallback"
        result["steps"] = [
            {
                "name": "自主模式入口判断",
                "tool": "goal_router",
                "reason": "Goal Manager 判断输入不适合任务级循环。",
                "status": "success",
                "summary": f"已回退普通问答：{reason}",
                "elapsed_ms": 0,
                "error": "",
            },
            *result.get("steps", []),
        ]
        return result

    return agent_runtime.run_agent_pro(
        user_input,
        use_web=True,
        top_k=3,
        web_max_results=2,
        preferred_sources=preferred_sources,
        router_mode=router_mode,
        source_strategy=source_strategy,
    )


def timeout_result(case: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "planner_mode": "timeout",
        "answer": "",
        "sources": [],
        "steps": [
            {
                "name": "Eval 超时保护",
                "tool": "eval_timeout",
                "reason": f"Case {case['case_id']} 超过单条用例时间限制。",
                "status": "failed",
                "summary": message,
                "elapsed_ms": 0,
                "error": message,
            }
        ],
        "tasks": [],
        "stop_reason": "eval_timeout",
        "error": message,
    }


def error_result(case: dict[str, Any], error: Exception) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    return {
        "planner_mode": "error",
        "answer": "",
        "sources": [],
        "steps": [
            {
                "name": "Eval 异常捕获",
                "tool": "eval_error",
                "reason": f"Case {case['case_id']} 执行时抛出异常。",
                "status": "failed",
                "summary": message,
                "elapsed_ms": 0,
                "error": message,
            }
        ],
        "tasks": [],
        "stop_reason": "eval_error",
        "error": message,
    }


def run_case_safely(
    case: dict[str, Any],
    case_timeout: int,
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
) -> dict[str, Any]:
    if case_timeout <= 0:
        try:
            return run_case(case, router_mode=router_mode, source_strategy=source_strategy)
        except Exception as error:
            return error_result(case, error)

    def handle_timeout(signum, frame):
        raise TimeoutError(f"超过 {case_timeout} 秒未完成")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(case_timeout)
    try:
        return run_case(case, router_mode=router_mode, source_strategy=source_strategy)
    except TimeoutError as error:
        return timeout_result(case, str(error))
    except Exception as error:
        return error_result(case, error)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_case_child(case: dict[str, Any], queue: Any, router_mode: str, source_strategy: str) -> None:
    try:
        queue.put({"ok": True, "result": run_case(case, router_mode=router_mode, source_strategy=source_strategy)})
    except Exception as error:
        queue.put({"ok": False, "result": error_result(case, error)})


def run_case_isolated(
    case: dict[str, Any],
    case_timeout: int,
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=run_case_child, args=(case, queue, router_mode, source_strategy))
    process.start()
    process.join(case_timeout if case_timeout > 0 else None)

    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return timeout_result(case, f"隔离子进程超过 {case_timeout} 秒未完成，已终止。")

    if queue.empty():
        return error_result(case, RuntimeError(f"隔离子进程退出但未返回结果，exitcode={process.exitcode}"))

    payload = queue.get()
    return payload["result"]


def actual_tools(result: dict[str, Any]) -> list[str]:
    return [step.get("tool", "") for step in result.get("steps", [])]


def actual_tasks(result: dict[str, Any]) -> list[str]:
    return [task.id for task in result.get("tasks", [])]


def score_expected_mode(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_mode")
    if not expected:
        return {"passed": True, "expected": "", "actual": result.get("planner_mode", "")}
    actual = result.get("planner_mode", "")
    return {"passed": actual == expected, "expected": expected, "actual": actual}


def score_expected_tools(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tools = actual_tools(result)
    missing = [tool for tool in case.get("expected_tools", []) if tool not in tools]
    return {"passed": not missing, "missing": missing, "actual": tools}


def score_forbidden_tools(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tools = actual_tools(result)
    violated = [tool for tool in case.get("forbidden_tools", []) if tool in tools]
    return {"passed": not violated, "violated": violated, "actual": tools}


def score_sources(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    sources = result.get("sources", [])
    source_types = [source.get("source_type", "unknown") for source in sources]
    missing_expected = [
        source_type for source_type in case.get("expected_sources", [])
        if source_type not in source_types
    ]
    forbidden = set(case.get("forbidden_sources", []))
    violations = [
        source for source in sources
        if source.get("source_type", "unknown") in forbidden
    ]
    return {
        "passed": not missing_expected and not violations,
        "source_types": source_types,
        "missing_expected": missing_expected,
        "violations": violations,
    }


def score_required_tasks(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tasks = actual_tasks(result)
    missing = [task for task in case.get("required_tasks", []) if task not in tasks]
    forbidden = [task for task in case.get("forbidden_tasks", []) if task in tasks]
    return {"passed": not missing and not forbidden, "missing": missing, "forbidden": forbidden, "actual": tasks}


def score_task_completion(result: dict[str, Any]) -> dict[str, Any]:
    tasks = result.get("tasks", [])
    if not tasks:
        return {"passed": True, "completion_rate": None}
    completed = [task for task in tasks if task.status == "completed"]
    return {
        "passed": len(completed) == len(tasks),
        "completion_rate": len(completed) / len(tasks),
        "completed": len(completed),
        "total": len(tasks),
    }


def score_stop_reason(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_stop_reason")
    if not expected:
        return {"passed": True, "expected": "", "actual": result.get("stop_reason", "")}
    actual = result.get("stop_reason", "")
    return {"passed": actual == expected, "expected": expected, "actual": actual}


def score_answer(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer", "")
    issues = []
    min_chars = case.get("min_answer_chars")
    if min_chars and len(answer) < min_chars:
        issues.append(f"答案长度少于 {min_chars} 字")
    for phrase in case.get("required_phrases", []):
        if phrase not in answer:
            issues.append(f"缺少必需短语：{phrase}")
    for phrase in case.get("expected_answer_phrases", []):
        if phrase not in answer:
            issues.append(f"缺少期望短语：{phrase}")
    for phrase in case.get("forbidden_answer_phrases", []):
        if phrase in answer:
            issues.append(f"出现禁止短语：{phrase}")
    return {"passed": not issues, "issues": issues, "answer_preview": answer[:220]}


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "expected_mode": score_expected_mode(case, result),
        "expected_tools": score_expected_tools(case, result),
        "forbidden_tools": score_forbidden_tools(case, result),
        "sources": score_sources(case, result),
        "required_tasks": score_required_tasks(case, result),
        "task_completion": score_task_completion(result),
        "stop_reason": score_stop_reason(case, result),
        "answer": score_answer(case, result),
    }
    passed = all(check["passed"] for check in checks.values())
    failed_checks = [name for name, check in checks.items() if not check["passed"]]
    return {
        "case_id": case["case_id"],
        "category": case.get("category", "unknown"),
        "passed": passed,
        "failed_checks": failed_checks,
        "checks": checks,
        "result": {
            "planner_mode": result.get("planner_mode", ""),
            "tools": actual_tools(result),
            "tasks": actual_tasks(result),
            "stop_reason": result.get("stop_reason", ""),
            "sources": result.get("sources", []),
            "answer": result.get("answer", ""),
        },
    }


def build_expected_behavior(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_mode": case.get("expected_mode", ""),
        "expected_tools": case.get("expected_tools", []),
        "forbidden_tools": case.get("forbidden_tools", []),
        "expected_sources": case.get("expected_sources", []),
        "forbidden_sources": case.get("forbidden_sources", []),
        "preferred_sources": case.get("preferred_sources", []),
        "required_tasks": case.get("required_tasks", []),
        "forbidden_tasks": case.get("forbidden_tasks", []),
        "success_criteria": case.get("success_criteria", []),
        "required_phrases": case.get("required_phrases", []),
        "forbidden_answer_phrases": case.get("forbidden_answer_phrases", []),
    }


def build_reference_context(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    references = []
    for source in evaluation["result"].get("sources", []):
        references.append({
            "source_type": source.get("source_type", "unknown"),
            "source": source.get("source", ""),
            "url": source.get("url", ""),
            "document": source.get("document", "")[:1200],
            "final_score": source.get("final_score", ""),
        })
    return references


def build_judge_payload(case: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    reference_context = build_reference_context(evaluation)
    if case.get("category") == "upload_status":
        preferred_sources = case.get("preferred_sources", [])
        reference_context.append({
            "source_type": "app_state",
            "source": "current_uploaded_sources",
            "document": (
                "当前已入库上传资料："
                + ("、".join(preferred_sources) if preferred_sources else "无")
            ),
        })

    return {
        "case_id": case["case_id"],
        "category": case.get("category", "unknown"),
        "user_prompt": case["user_input"],
        "agent_result": evaluation["result"].get("answer", ""),
        "reference_context": reference_context,
        "tool_trace": evaluation["result"].get("tools", []),
        "task_trace": evaluation["result"].get("tasks", []),
        "expected_behavior": build_expected_behavior(case),
        "rule_result": {
            "pass": evaluation["passed"],
            "failed_checks": evaluation["failed_checks"],
            "checks": evaluation["checks"],
        },
        "rubric": JUDGE_RUBRIC,
        "output_schema": JUDGE_OUTPUT_SCHEMA,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    stripped = stripped.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            snippet = stripped[start:end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                sanitized = re.sub(r",\s*([}\]])", r"\1", snippet)
                sanitized = re.sub(r"(?<!\\)\n", r"\\n", sanitized)
                return json.loads(sanitized)
        raise


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def aggregate_judge_scores(category: str, judge_result: dict[str, Any]) -> dict[str, Any]:
    raw_scores = judge_result.get("scores", {})
    scores = {name: normalize_score(value) for name, value in raw_scores.items()}
    weights = JUDGE_WEIGHTS_BY_CATEGORY.get(category, DEFAULT_JUDGE_WEIGHTS)
    overall = sum(scores.get(name, 0.0) * weight for name, weight in weights.items())

    hard_fail_reasons = []
    if category in {"web_rag", "document_qa", "source_scope"} and scores.get("groundedness", 5.0) < 3.0:
        hard_fail_reasons.append("groundedness 低于硬门槛")
    if category in {"document_qa", "source_scope", "upload_status"} and scores.get("source_usage", 5.0) < 3.0:
        hard_fail_reasons.append("source_usage 低于硬门槛")
    if scores.get("safety", 5.0) < 3.0:
        hard_fail_reasons.append("safety 低于硬门槛")

    passed = overall >= JUDGE_PASS_THRESHOLD and not hard_fail_reasons
    return {
        "scores": scores,
        "overall_score": round(overall, 2),
        "pass": passed,
        "hard_fail_reasons": hard_fail_reasons,
    }


def call_llm_judge(payload: dict[str, Any]) -> dict[str, Any]:
    client = agent_runtime.agent.get_deepseek_client()
    if client is None:
        return {
            "enabled": True,
            "available": False,
            "pass": False,
            "overall_score": 0,
            "scores": {},
            "failed_dimensions": ["judge_client"],
            "reason": "未找到 DEEPSEEK_API_KEY，无法调用 LLM-as-Judge。",
        }

    messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    request_args = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "timeout": agent_runtime.agent.LLM_TIMEOUT_SECONDS,
    }
    try:
        response = client.chat.completions.create(**request_args)
    except TypeError:
        request_args.pop("response_format", None)
        response = client.chat.completions.create(**request_args)
    raw_text = response.choices[0].message.content or "{}"
    try:
        parsed = extract_json_object(raw_text)
    except json.JSONDecodeError:
        retry_messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "上一条评估输出不是合法 JSON。请只把它修正为合法 JSON，"
                    "不得添加 Markdown 或解释。\n\n原始输出：\n" + raw_text
                ),
            },
        ]
        retry_args = {
            "model": JUDGE_MODEL,
            "messages": retry_messages,
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "timeout": agent_runtime.agent.LLM_TIMEOUT_SECONDS,
        }
        try:
            retry_response = client.chat.completions.create(**retry_args)
        except TypeError:
            retry_args.pop("response_format", None)
            retry_response = client.chat.completions.create(**retry_args)
        parsed = extract_json_object(retry_response.choices[0].message.content or "{}")

    if not isinstance(parsed.get("scores"), dict) or not parsed.get("scores"):
        retry_messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请重新完成这次 Agent 评估。必须输出包含 scores、overall_score、pass、"
                    "failed_dimensions、reason 的合法 JSON，scores 不能为空。\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ]
        retry_args = {
            "model": JUDGE_MODEL,
            "messages": retry_messages,
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "timeout": agent_runtime.agent.LLM_TIMEOUT_SECONDS,
        }
        try:
            retry_response = client.chat.completions.create(**retry_args)
        except TypeError:
            retry_args.pop("response_format", None)
            retry_response = client.chat.completions.create(**retry_args)
        parsed = extract_json_object(retry_response.choices[0].message.content or "{}")

    if not isinstance(parsed.get("scores"), dict) or not parsed.get("scores"):
        parsed = {
            "scores": {
                "task_success": 0,
                "groundedness": 0,
                "source_usage": 0,
                "completeness": 0,
                "clarity": 0,
                "safety": 0,
            },
            "overall_score": 0,
            "pass": False,
            "failed_dimensions": ["judge_invalid_output"],
            "reason": "Judge 返回合法 JSON，但缺少必需的 scores 字段。",
        }
    aggregation = aggregate_judge_scores(payload["category"], parsed)
    return {
        "enabled": True,
        "available": True,
        "model": JUDGE_MODEL,
        "pass": aggregation["pass"],
        "overall_score": aggregation["overall_score"],
        "scores": aggregation["scores"],
        "hard_fail_reasons": aggregation["hard_fail_reasons"],
        "model_pass": bool(parsed.get("pass", False)),
        "model_overall_score": normalize_score(parsed.get("overall_score", 0)),
        "failed_dimensions": parsed.get("failed_dimensions", []),
        "reason": parsed.get("reason", ""),
    }


def attach_judge_result(case: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    payload = build_judge_payload(case, evaluation)
    try:
        judge = call_llm_judge(payload)
    except Exception as error:
        judge = {
            "enabled": True,
            "available": False,
            "pass": False,
            "overall_score": 0,
            "scores": {},
            "failed_dimensions": ["judge_error"],
            "reason": f"{type(error).__name__}: {error}",
        }

    rule_pass = evaluation["passed"]
    evaluation["rule_pass"] = rule_pass
    evaluation["judge"] = judge
    evaluation["passed"] = rule_pass and judge["pass"]
    if not evaluation["passed"] and "judge" not in evaluation["failed_checks"] and not judge["pass"]:
        evaluation["failed_checks"].append("judge")
    return evaluation


def run_eval(
    cases: list[dict[str, Any]],
    mode: str = "mock",
    case_timeout: int = 120,
    isolate_cases: bool = False,
    judge: bool = False,
    router_mode: str = ROUTER_MODE_RULES,
    source_strategy: str = SOURCE_STRATEGY_AUTO,
    stable_web: bool = False,
) -> dict[str, Any]:
    original_tools = None
    if mode == "mock":
        original_tools = install_fake_tools()
    elif stable_web:
        original_tools = install_stable_web_tool()
    rows = []
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] running {case['case_id']} ({case.get('category', 'unknown')})")
            if isolate_cases:
                result = run_case_isolated(
                    case,
                    case_timeout,
                    router_mode=router_mode,
                    source_strategy=source_strategy,
                )
            else:
                result = run_case_safely(
                    case,
                    case_timeout,
                    router_mode=router_mode,
                    source_strategy=source_strategy,
                )
            evaluation = evaluate_case(case, result)
            if judge:
                evaluation = attach_judge_result(case, evaluation)
            rows.append({
                "case": case,
                "evaluation": evaluation,
            })
    finally:
        if original_tools is not None:
            restore_tools(original_tools)

    total = len(rows)
    passed = sum(1 for row in rows if row["evaluation"]["passed"])
    by_category = defaultdict(lambda: {"total": 0, "passed": 0})
    for row in rows:
        category = row["evaluation"]["category"]
        by_category[category]["total"] += 1
        by_category[category]["passed"] += int(row["evaluation"]["passed"])

    failed_check_counter = Counter()
    for row in rows:
        failed_check_counter.update(row["evaluation"]["failed_checks"])

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0,
        "mode": mode,
        "judge_enabled": judge,
        "judge_model": JUDGE_MODEL if judge else "",
        "router_mode": router_mode,
        "source_strategy": source_strategy,
        "stable_web": stable_web,
        "by_category": dict(by_category),
        "failed_check_counts": dict(failed_check_counter),
        "rows": rows,
    }


def esc(value: Any) -> str:
    return html.escape(str(value))


def render_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    category_rows = "\n".join(
        f"<tr><td>{esc(category)}</td><td>{stats['passed']}/{stats['total']}</td><td>{stats['passed'] / stats['total']:.0%}</td></tr>"
        for category, stats in sorted(report["by_category"].items())
    )
    failed_checks = report["failed_check_counts"] or {}
    failed_check_rows = "\n".join(
        f"<tr><td>{esc(name)}</td><td>{count}</td></tr>"
        for name, count in sorted(failed_checks.items())
    ) or "<tr><td>无</td><td>0</td></tr>"

    case_rows = []
    for row in report["rows"]:
        case = row["case"]
        evaluation = row["evaluation"]
        status = "通过" if evaluation["passed"] else "失败"
        status_class = "pass" if evaluation["passed"] else "fail"
        checks_summary = "\n".join(
            f"{name}: {'通过' if check['passed'] else '失败'}"
            for name, check in evaluation["checks"].items()
        )
        judge = evaluation.get("judge", {})
        if judge:
            score_lines = "\n".join(
                f"{name}: {score:.1f}"
                for name, score in sorted(judge.get("scores", {}).items())
            )
            judge_summary = (
                f"Judge: {'通过' if judge.get('pass') else '失败'}\n"
                f"总分: {judge.get('overall_score', 0)}\n"
                f"{score_lines}\n"
                f"硬门槛: {', '.join(judge.get('hard_fail_reasons', [])) or '无'}\n"
                f"理由: {judge.get('reason', '')}"
            )
        else:
            judge_summary = "未启用"
        actual = evaluation["result"]
        case_rows.append(f"""
        <tr>
          <td><strong>{esc(case['case_id'])}</strong><br><span>{esc(case.get('category', ''))}</span></td>
          <td>{esc(case['user_input'])}</td>
          <td class="{status_class}">{status}</td>
          <td><pre>{esc(checks_summary)}</pre></td>
          <td><pre>{esc(judge_summary)}</pre></td>
          <td><pre>{esc(json.dumps(actual, ensure_ascii=False, indent=2)[:1600])}</pre></td>
        </tr>
        """)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Eval 报告</title>
  <style>
    body {{ margin: 0; background: #f6f7fb; color: #202431; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 40px 24px 72px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 32px 0 14px; font-size: 22px; }}
    .sub {{ color: #667085; margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #e5e7ef; border-radius: 8px; padding: 18px; }}
    .metric {{ font-size: 30px; font-weight: 760; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #dc2626; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7ef; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #e5e7ef; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #f1f5f9; }}
    tr:last-child td {{ border-bottom: none; }}
    pre {{ margin: 0; white-space: pre-wrap; font-size: 12px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; max-height: 260px; overflow: auto; }}
    .note {{ border-left: 4px solid #2563eb; background: #eff6ff; color: #1e3a8a; padding: 12px 14px; border-radius: 6px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 28px 16px 56px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Agent Eval 报告</h1>
  <div class="sub">运行模式：{esc(report.get('mode', 'mock'))}。路由模式：{esc(report.get('router_mode', 'rules'))}。资料策略：{esc(report.get('source_strategy', 'auto'))}。稳定网页夹具：{esc('已启用' if report.get('stable_web') else '未启用')}。Judge：{esc('已启用 ' + report.get('judge_model', '') if report.get('judge_enabled') else '未启用')}。规则检查系统是否跑对；LLM-as-Judge 检查语义质量、资料忠实度和来源使用。</div>

  <section class="grid">
    <div class="card"><div class="metric">{report['total']}</div><div>总样本</div></div>
    <div class="card"><div class="metric pass">{report['passed']}</div><div>通过</div></div>
    <div class="card"><div class="metric fail">{report['failed']}</div><div>失败</div></div>
    <div class="card"><div class="metric">{report['pass_rate']:.0%}</div><div>通过率</div></div>
  </section>

  <h2>分场景通过率</h2>
  <table><thead><tr><th>场景</th><th>通过 / 总数</th><th>通过率</th></tr></thead><tbody>{category_rows}</tbody></table>

  <h2>失败检查项</h2>
  <table><thead><tr><th>检查项</th><th>失败次数</th></tr></thead><tbody>{failed_check_rows}</tbody></table>

  <h2>逐 Case 明细</h2>
  <table>
    <thead><tr><th>Case</th><th>输入</th><th>最终结果</th><th>规则检查</th><th>Judge 语义评分</th><th>实际输出摘要</th></tr></thead>
    <tbody>{''.join(case_rows)}</tbody>
  </table>

  <h2>结论</h2>
  <div class="note">这版 Eval 同时支持规则校验和 LLM-as-Judge。规则负责硬边界，Judge 负责语义质量；最终结果要求两者都通过。</div>
</main>
</body>
</html>"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--suite", default="", help="只运行指定 suite，例如 smoke、regression、benchmark。")
    parser.add_argument("--case-id", action="append", default=[], help="只运行指定 case_id，可重复传入。")
    parser.add_argument("--case-timeout", type=int, default=120, help="单条 case 的超时时间，单位秒；<=0 表示不限制。")
    parser.add_argument("--isolate-cases", action="store_true", help="每条 case 使用隔离子进程执行，适合真实 API benchmark。")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge 语义质量评分。")
    parser.add_argument("--stable-web", action="store_true", help="真实模型评估时使用稳定网页夹具，避免实时搜索源波动。")
    parser.add_argument("--router-mode", choices=sorted(ROUTER_MODES), default=ROUTER_MODE_RULES)
    parser.add_argument("--source-strategy", choices=sorted(SOURCE_STRATEGIES), default=SOURCE_STRATEGY_AUTO)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.suite:
        cases = [case for case in cases if args.suite in case.get("suite", [])]
    if args.case_id:
        case_ids = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in case_ids]

    if args.mode == "real":
        ensure_eval_upload_fixtures(cases)

    report = run_eval(
        cases,
        mode=args.mode,
        case_timeout=args.case_timeout,
        isolate_cases=args.isolate_cases,
        judge=args.judge,
        router_mode=args.router_mode,
        source_strategy=args.source_strategy,
        stable_web=args.stable_web,
    )
    render_report(report, Path(args.report))

    print(f"Mode: {report['mode']}")
    print(f"Total: {report['total']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Pass rate: {report['pass_rate']:.0%}")
    print(f"Judge: {'enabled' if report['judge_enabled'] else 'disabled'}")
    print(f"Router mode: {report['router_mode']}")
    print(f"Source strategy: {report['source_strategy']}")
    print(f"Stable web: {'enabled' if report['stable_web'] else 'disabled'}")
    print(f"Report: {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
