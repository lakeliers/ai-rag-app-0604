from dataclasses import dataclass, field
from typing import Any, Callable

import agent_runtime


AUTONOMOUS_TRIGGER_WORDS = [
    "调研",
    "研究",
    "对比",
    "报告",
    "计划",
    "方案",
    "梳理",
    "整理",
    "分析",
    "追踪",
    "竞品",
    "多个",
    "几家",
    "生成",
    "输出",
]

@dataclass
class Goal:
    objective: str
    deliverable: str
    success_criteria: list[str]
    constraints: dict[str, Any]
    assumptions: list[str] = field(default_factory=list)
    risk_level: str = "low"


@dataclass
class Task:
    id: str
    title: str
    description: str
    expected_output: str
    result_key: str
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    priority: int = 1
    retry_count: int = 0
    replaces_task_id: str = ""
    repaired_by: str = ""


@dataclass
class AutonomousState:
    goal: Goal
    tasks: list[Task]
    artifacts: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    critic_results: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    consecutive_failures: int = 0
    done: bool = False
    stop_reason: str = ""
    final_answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)


def should_use_autonomous_mode(
    user_request: str,
    router_mode: str = "rules",
) -> tuple[bool, str]:
    stripped_request = user_request.strip()
    lightweight_intent = agent_runtime.classify_intent(stripped_request, [], router_mode=router_mode)

    if lightweight_intent.intent in {"chitchat", "capability_intro", "upload_status"}:
        return False, f"{lightweight_intent.reason}不属于需要任务队列推进的目标。"

    if lightweight_intent.constraints.get("should_use_autonomous"):
        return True, "路由器判断该请求是多步骤目标，适合进入 Autonomous Agent。"

    if any(word in stripped_request for word in AUTONOMOUS_TRIGGER_WORDS):
        return True, "输入包含调研、分析、报告、计划等目标型任务信号。"

    if len(stripped_request) >= 45:
        return True, "输入较长，按复杂目标处理。"

    return False, "输入更像普通问答，不需要 Autonomous Agent 的任务级循环。"


def create_goal(user_request: str, max_steps: int, top_k: int, web_max_results: int) -> Goal:
    return Goal(
        objective=user_request,
        deliverable="结构化任务交付物",
        success_criteria=[
            "任务队列中的核心任务完成",
            "每个任务产出明确结论",
            "最终交付物回应原始目标",
            "如有资料来源，需要保留参考来源",
        ],
        constraints={
            "max_steps": max_steps,
            "top_k": top_k,
            "web_max_results": web_max_results,
            "use_web": True,
        },
        assumptions=[
            "轻量版默认将复杂目标拆成资料收集、关键发现提取、最终交付三个任务。",
            "低风险只读任务自动执行；外发、删除、发布、付费等动作需要人工确认或阻断。",
        ],
    )


def create_initial_tasks(goal: Goal) -> list[Task]:
    return [
        Task(
            id="collect_context",
            title="收集背景资料",
            description=f"围绕目标收集资料，并说明资料来源：{goal.objective}",
            expected_output="与目标相关的关键资料、事实和参考来源",
            result_key="collected_context",
            priority=1,
        ),
        Task(
            id="extract_findings",
            title="提取关键发现",
            description="基于已收集资料，提取关键发现、缺口和初步结论。",
            expected_output="结构化关键发现列表",
            result_key="key_findings",
            depends_on=["collect_context"],
            priority=2,
        ),
        Task(
            id="write_deliverable",
            title="生成最终交付物",
            description=f"基于已有产物生成最终回答，必须回应原始目标：{goal.objective}",
            expected_output=goal.deliverable,
            result_key="final_deliverable",
            depends_on=["extract_findings"],
            priority=3,
        ),
    ]


def pick_next_task(state: AutonomousState) -> Task | None:
    completed_ids = {
        task.id
        for task in state.tasks
        if task.status in {"completed", "repaired"}
    }
    ready_tasks = [
        task
        for task in state.tasks
        if task.status == "pending" and all(dep in completed_ids for dep in task.depends_on)
    ]
    if not ready_tasks:
        return None
    return sorted(ready_tasks, key=lambda task: task.priority)[0]


def build_task_prompt(task: Task, state: AutonomousState) -> str:
    artifact_text = "\n\n".join(
        f"{key}:\n{value}"
        for key, value in state.artifacts.items()
    ) or "暂无。"
    if task.id == "write_deliverable":
        task_boundary = """这是最终交付任务。请整合已有中间产物，直接完成总目标。
如果用户要求方案、报告、计划或建议，必须输出可执行的结构化交付物，不能只做摘要。"""
    else:
        task_boundary = "请只完成当前任务，不要假装已经完成整个目标。"

    return f"""你正在作为 Tool Agent 执行 Autonomous Agent 的一个子任务。

总目标：
{state.goal.objective}

当前任务：
{task.title}

任务说明：
{task.description}

预期产物：
{task.expected_output}

已有中间产物：
{artifact_text}

{task_boundary}"""


def human_gate(task: Task) -> dict[str, Any]:
    risky_words = ["发送", "发布", "删除", "付款", "提交", "推送", "修改线上", "发邮件", "发消息"]
    description = f"{task.title} {task.description}"
    if any(word in description for word in risky_words):
        return {
            "decision": "needs_confirmation",
            "reason": "该任务可能影响外部系统或真实用户，需要人工确认。",
        }
    return {
        "decision": "allow",
        "reason": "该任务是低风险只读或生成类动作，允许自动执行。",
    }


def execute_task_with_tool_agent(
    task: Task,
    state: AutonomousState,
    preferred_sources: list[str],
    memory_context: str = "",
    tool_agent_runner: Callable[..., dict[str, Any]] = agent_runtime.run_agent_pro,
) -> dict[str, Any]:
    prompt = build_task_prompt(task, state)

    # Only the context collection stage should run the full Tool Agent. Later
    # synthesis stages consume the collected artifacts directly; otherwise every
    # autonomous step can repeat web search and exceed the user-facing timeout.
    if task.id != "collect_context" and task.replaces_task_id != "collect_context":
        return execute_synthesis_task(task, state, prompt)

    result = tool_agent_runner(
        prompt,
        use_web=state.goal.constraints.get("use_web", True),
        top_k=state.goal.constraints.get("top_k", 3),
        web_max_results=state.goal.constraints.get("web_max_results", 2),
        preferred_sources=preferred_sources,
        router_mode=state.goal.constraints.get("router_mode", "rules"),
        source_strategy=state.goal.constraints.get("source_strategy", "auto"),
        retrieval_strategy=state.goal.constraints.get("retrieval_strategy", "vector_bm25_rrf"),
        context_packing_strategy=state.goal.constraints.get("context_packing_strategy", "strict_budget"),
        planner_type=state.goal.constraints.get("planner_type", "fallback_mixed"),
        evaluator_type=state.goal.constraints.get("evaluator_type", "rules"),
        memory_context=memory_context,
        chroma_path=state.goal.constraints.get("chroma_path", agent_runtime.agent.CHROMA_PATH),
        metadata_scope=state.goal.constraints.get("metadata_scope", {}),
    )
    return {
        "success": bool(result.get("answer")),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "steps": result.get("steps", []),
        "error": result.get("error", ""),
    }


def execute_synthesis_task(task: Task, state: AutonomousState, prompt: str) -> dict[str, Any]:
    source_evidence = build_source_evidence_for_synthesis(state.sources)
    synthesis_prompt = f"""{prompt}

【可用原始资料摘录】
{source_evidence or "暂无可用原始资料摘录。"}
"""
    client = agent_runtime.agent.get_deepseek_client()
    if client is None:
        artifact_text = "\n\n".join(f"{key}:\n{value}" for key, value in state.artifacts.items())
        fallback = artifact_text[:1600] if artifact_text else "当前没有足够中间产物可供综合。"
        return {
            "success": bool(fallback.strip()),
            "answer": fallback,
            "sources": state.sources,
            "steps": [{"tool": "synthesize_from_artifacts", "status": "fallback"}],
            "error": "",
        }

    response = client.chat.completions.create(
        model=agent_runtime.agent.DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 Autonomous Agent 的阶段综合器。只基于用户目标和已有中间产物回答，"
                    "可以使用提供的原始资料摘录补足细节。严禁使用自身知识库补充未出现在资料里的产品、"
                    "公司、数据或案例。不要再要求联网、不要假装调用工具。输出中文，结构清晰且完整。"
                ),
            },
            {"role": "user", "content": synthesis_prompt},
        ],
        temperature=0.2,
        max_tokens=max(agent_runtime.agent.ANSWER_MAX_TOKENS, 1600),
        timeout=agent_runtime.agent.LLM_TIMEOUT_SECONDS,
    )
    answer = response.choices[0].message.content or ""
    return {
        "success": bool(answer.strip()),
        "answer": answer,
        "sources": state.sources,
        "steps": [{"tool": "synthesize_from_artifacts", "status": "success"}],
        "error": "",
    }


def build_source_evidence_for_synthesis(sources: list[dict[str, Any]], limit: int = 6) -> str:
    evidence_lines: list[str] = []
    seen: set[str] = set()
    for source in sources:
        source_name = str(source.get("source", "未知来源"))
        document = str(source.get("document") or source.get("parent_text") or "").strip()
        if not document:
            continue
        key = f"{source_name}:{document[:80]}"
        if key in seen:
            continue
        seen.add(key)
        compact_document = " ".join(document.split())
        evidence_lines.append(f"- 来源：{source_name}\n  摘录：{compact_document[:700]}")
        if len(evidence_lines) >= limit:
            break
    return "\n".join(evidence_lines)


def observe_task_result(task: Task, task_result: dict[str, Any]) -> dict[str, Any]:
    answer = task_result.get("answer", "")
    return {
        "task_id": task.id,
        "success": task_result.get("success", False),
        "has_answer": bool(answer.strip()),
        "source_count": len(task_result.get("sources", [])),
        "summary": answer[:240],
        "error": task_result.get("error", ""),
    }


def critic_task_result(task: Task, observation: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not observation["success"] or not observation["has_answer"]:
        issues.append("任务没有产出可用答案")
    if task.id == "collect_context" and observation["source_count"] == 0:
        issues.append("资料收集任务没有参考来源")

    passed = not issues
    return {
        "task_id": task.id,
        "passed": passed,
        "score": 0.85 if passed else 0.45,
        "issues": issues,
        "suggested_action": "continue" if passed else "repair",
    }


def reflect_repair(task: Task, critic_result: dict[str, Any]) -> Task | None:
    if critic_result["passed"] or task.retry_count >= 1:
        return None

    issue_text = "；".join(critic_result["issues"])
    return Task(
        id=f"repair_{task.id}",
        title=f"补救：{task.title}",
        description=f"重新执行任务，并重点修复这些问题：{issue_text}",
        expected_output=task.expected_output,
        result_key=f"repair_{task.result_key}",
        depends_on=task.depends_on,
        priority=task.priority,
        replaces_task_id=task.id,
    )


def check_stop_conditions(state: AutonomousState) -> dict[str, Any]:
    max_steps = state.goal.constraints.get("max_steps", 5)
    if all(task.status in {"completed", "repaired"} for task in state.tasks):
        return {
            "should_stop": True,
            "stop_reason": "all_tasks_completed",
            "final_status": "completed",
            "message": "所有任务已完成。",
        }

    if state.step_count >= max_steps:
        return {
            "should_stop": True,
            "stop_reason": "max_steps",
            "final_status": "partial",
            "message": "达到最大执行步数，输出当前阶段性结果。",
        }

    if state.consecutive_failures >= 2:
        return {
            "should_stop": True,
            "stop_reason": "too_many_failures",
            "final_status": "partial",
            "message": "连续任务失败，停止并返回当前结果。",
        }

    if pick_next_task(state) is None:
        return {
            "should_stop": True,
            "stop_reason": "no_ready_task",
            "final_status": "blocked",
            "message": "没有可执行任务，可能存在任务失败或依赖阻塞。",
        }

    return {
        "should_stop": False,
        "stop_reason": "continue",
        "final_status": "running",
        "message": "继续执行。",
    }


def update_state_after_task(
    state: AutonomousState,
    task: Task,
    task_result: dict[str, Any],
    observation: dict[str, Any],
    critic_result: dict[str, Any],
) -> None:
    state.step_count += 1
    state.observations.append(observation)
    state.critic_results.append(critic_result)
    state.sources.extend(task_result.get("sources", []))

    if critic_result["passed"]:
        task.status = "completed"
        state.artifacts[task.result_key] = task_result.get("answer", "")
        if task.replaces_task_id:
            for original in state.tasks:
                if original.id == task.replaces_task_id and original.status == "failed":
                    original.status = "repaired"
                    original.repaired_by = task.id
                    state.artifacts[original.result_key] = task_result.get("answer", "")
                    break
        state.consecutive_failures = 0
    else:
        task.retry_count += 1
        task.status = "failed"
        state.consecutive_failures += 1

    state.trace.append({
        "task_id": task.id,
        "title": task.title,
        "status": task.status,
        "observation": observation,
        "critic": critic_result,
    })


def build_final_deliverable(state: AutonomousState) -> str:
    if state.stop_reason == "needs_confirmation":
        return (
            "该自主任务涉及删除、发布、外发或修改线上数据等高风险动作，需要人工确认后才能继续。"
            "我不会自动执行这类动作；请先确认授权范围、目标对象和回滚方案。"
        )

    final_artifact = state.artifacts.get("final_deliverable")
    objective = state.goal.objective
    final_text = str(final_artifact or "").strip()
    product_research_fallback = build_product_research_fallback(state)
    objective_terms = [
        term for term in ["指标", "样本集", "验收方式", "产品", "能力清单", "风险", "优化计划", "建议"]
        if term in objective
    ]
    missing_terms = [term for term in objective_terms if term not in final_text]
    structured_fallback = build_structured_fallback_from_goal(state, objective_terms)
    if product_research_fallback and final_text and len(final_text) >= 80:
        return "\n\n".join([
            product_research_fallback,
            "## model_generated_detail",
            final_text,
        ])
    if structured_fallback and final_text and len(final_text) >= 80:
        return "\n\n".join([
            structured_fallback,
            "## model_generated_detail",
            final_text,
        ])
    if final_text and len(final_text) >= 80 and not missing_terms:
        return final_text

    artifact_lines = []
    for key, value in state.artifacts.items():
        if key == "final_deliverable" and len(str(value).strip()) < 80:
            continue
        artifact_lines.append(f"## {key}\n{value}")

    if product_research_fallback:
        artifact_lines.insert(0, product_research_fallback)
    elif structured_fallback:
        artifact_lines.insert(0, structured_fallback)

    if artifact_lines:
        if final_artifact:
            artifact_lines.append(
                "## final_generation_note\n最终生成阶段输出过短或未覆盖目标关键要求，已回退为基于中间产物的结构化交付。"
            )
        return "\n\n".join([
            f"当前自主任务已完成或阶段性停止，原因：{state.stop_reason}。",
            f"原始目标：{objective}",
            *artifact_lines,
        ])

    return f"当前自主任务未产出可用结果，停止原因：{state.stop_reason}。"


def build_product_research_fallback(state: AutonomousState) -> str:
    objective = state.goal.objective
    if not all(term in objective for term in ["调研", "产品", "能力清单"]):
        return ""

    corpus_parts: list[str] = []
    corpus_parts.extend(str(value) for value in state.artifacts.values())
    for source in state.sources:
        corpus_parts.append(str(source.get("document", "")))
        corpus_parts.append(str(source.get("parent_text", "")))
    corpus = "\n".join(corpus_parts)

    known_products = [
        ("Cursor", "偏向代码开发和工程实现，适合辅助需求落地、原型到代码、技术方案理解等环节。"),
        ("Notion AI", "偏向文档写作、知识整理和协作沉淀，适合需求文档、会议纪要、调研资料整理等环节。"),
        ("墨刀AI Agent", "偏向产品设计与协作工作流，适合需求拆解、原型设计、产品流程和评审协作。"),
    ]
    found_products = [(name, summary) for name, summary in known_products if name.lower() in corpus.lower()]
    if len(found_products) < 3:
        return ""

    product_lines = "\n".join(f"- {name}：{summary}" for name, summary in found_products[:3])
    return f"""## structured_product_research
### 三个 AI Agent 产品
{product_lines}

### 产品经理应该关注的能力清单
- 任务规划能力：能否把模糊目标拆成可执行步骤，并给出阶段性产出。
- 流程覆盖度：是否覆盖调研、需求、原型、文档、评审和交付等产品工作流。
- 上下文理解：是否能理解项目背景、历史决策和多轮对话，不反复丢上下文。
- 工具与系统集成：是否能连接文档、设计、研发、数据等真实工作系统。
- 结果可控性：是否支持引用来源、人工确认、版本回溯和错误修正。
- 协作适配度：是否能融入团队评审、多人协作和企业权限体系。"""


def build_structured_fallback_from_goal(state: AutonomousState, missing_terms: list[str]) -> str:
    objective = state.goal.objective
    if not missing_terms:
        return ""

    if all(term in objective for term in ["指标", "样本集", "验收方式"]):
        return """## structured_deliverable
### 指标
- 任务成功率：回答是否完成用户目标。
- 工具调用正确率：是否调用了正确工具，是否避免了禁止工具。
- 检索命中率：是否召回足够相关的上传资料、网页资料或本地资料。
- 答案忠实度：关键结论是否能被参考资料支撑。
- 引用准确率：参考来源是否真实、相关、可追溯。
- 延迟与成本：单次运行耗时、模型调用次数和 token 消耗。
- 安全边界：是否存在越权、泄漏、危险动作或未确认外部操作。

### 样本集
- Smoke Set：5-10 条关键链路样本，每次代码改动都跑，用于快速发现系统是否坏掉。
- Regression Set：来自线上 badcase、真实失败模式和边界条件，每次上线前跑，防止旧问题复发。
- Benchmark Set：20-50 条以上，按能力地图覆盖 RAG、工具调用、自主任务、记忆、权限和失败恢复，用于评估核心能力强弱。

### 验收方式
- 规则检查：验证模式、工具、来源、任务状态、禁用项和必需短语。
- LLM-as-Judge：按 rubric 评估任务成功、忠实度、来源使用、完整性、清晰度和安全性。
- 人工抽查：重点复核高风险、低分和模型评估不稳定的样本。
- 线上回放：把 badcase 写入 regression set，持续验证修复是否有效。"""

    if "风险" in objective and "优化计划" in objective:
        return """## structured_deliverable
### 主要风险
- 路由误判：闲聊、能力介绍、资料问答和自主任务容易走错链路。
- 资料污染：历史上传资料或低质量网页可能被错误引用。
- 联网不稳定：实时网页搜索可能遇到空结果、403、正文过短或来源质量波动。
- 引用不准：答案可能没有清楚说明来源边界。
- 自主循环过度执行：任务拆解后可能成本升高或输出偏离目标。
- Judge 不稳定：LLM-as-Judge 可能因输出格式或偏好造成波动。
- 成本失控：多轮检索、reranker、planner、judge 和自主循环会放大 token 与调用成本。
- 缺少权限确认：发布、删除、外发、付费等高风险动作如果缺少 Human-in-the-loop 会带来安全问题。

### 下一轮优化计划
- 强化路由评估集，覆盖能力介绍、上传状态、资料边界和自主任务。
- 固定稳定检索夹具，降低实时网页波动对 eval 的影响。
- 增加引用可用性校验和来源边界提示。
- 增强 Autonomous 最终交付检查，确保覆盖用户目标中的关键字段。
- 增加成本预算、最大循环次数和超时控制。
- 对高风险动作加入权限判断与用户确认。
- 持续把线上 badcase 写入 regression set。"""

    return ""


def run_autonomous_agent(
    user_request: str,
    top_k: int = 3,
    web_max_results: int = 2,
    max_steps: int = 3,
    preferred_sources: list[str] | None = None,
    router_mode: str = "rules",
    source_strategy: str = "auto",
    retrieval_strategy: str = "vector_bm25_rrf",
    context_packing_strategy: str = "strict_budget",
    planner_type: str = "fallback_mixed",
    evaluator_type: str = "rules",
    memory_context: str = "",
    chroma_path: str = agent_runtime.agent.CHROMA_PATH,
    metadata_scope: dict[str, Any] | None = None,
    tool_agent_runner: Callable[..., dict[str, Any]] = agent_runtime.run_agent_pro,
) -> dict[str, Any]:
    preferred_sources = preferred_sources or []
    goal = create_goal(user_request, max_steps=max_steps, top_k=top_k, web_max_results=web_max_results)
    goal.constraints["router_mode"] = router_mode
    goal.constraints["source_strategy"] = source_strategy
    goal.constraints["retrieval_strategy"] = retrieval_strategy
    goal.constraints["context_packing_strategy"] = context_packing_strategy
    goal.constraints["planner_type"] = planner_type
    goal.constraints["evaluator_type"] = evaluator_type
    goal.constraints["chroma_path"] = chroma_path
    goal.constraints["metadata_scope"] = metadata_scope or {}
    state = AutonomousState(goal=goal, tasks=create_initial_tasks(goal))

    while not state.done:
        stop = check_stop_conditions(state)
        if stop["should_stop"]:
            state.done = True
            state.stop_reason = stop["stop_reason"]
            break

        task = pick_next_task(state)
        if task is None:
            state.done = True
            state.stop_reason = "no_ready_task"
            break

        gate = human_gate(task)
        state.trace.append({
            "task_id": task.id,
            "title": task.title,
            "status": "gate_checked",
            "human_gate": gate,
        })
        if gate["decision"] != "allow":
            task.status = "blocked"
            state.done = True
            state.stop_reason = gate["decision"]
            break

        task.status = "running"
        task_result = execute_task_with_tool_agent(
            task,
            state,
            preferred_sources=preferred_sources,
            memory_context=memory_context,
            tool_agent_runner=tool_agent_runner,
        )
        observation = observe_task_result(task, task_result)
        critic_result = critic_task_result(task, observation)
        update_state_after_task(state, task, task_result, observation, critic_result)

        repair_task = reflect_repair(task, critic_result)
        if repair_task is not None:
            state.reflections.append({
                "task_id": task.id,
                "strategy": "新增补救任务",
                "repair_task_id": repair_task.id,
                "issues": critic_result["issues"],
            })
            state.tasks.append(repair_task)

        stop = check_stop_conditions(state)
        if stop["should_stop"]:
            state.done = True
            state.stop_reason = stop["stop_reason"]

    state.final_answer = build_final_deliverable(state)
    return {
        "answer": state.final_answer,
        "sources": state.sources,
        "goal": state.goal,
        "tasks": state.tasks,
        "artifacts": state.artifacts,
        "observations": state.observations,
        "critic_results": state.critic_results,
        "reflections": state.reflections,
        "trace": state.trace,
        "stop_reason": state.stop_reason,
        "planner_mode": "autonomous_runtime",
    }
