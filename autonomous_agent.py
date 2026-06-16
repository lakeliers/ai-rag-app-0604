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

CHITCHAT_WORDS = [
    "你好",
    "您好",
    "嗨",
    "hello",
    "hi",
    "我是",
    "认识一下",
    "你是谁",
    "介绍一下你自己",
    "你能做什么",
    "你能做些什么",
    "你能做哪些事",
    "你会什么",
    "你擅长什么",
    "能帮我什么",
    "可以帮我什么",
    "能帮我做什么",
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


def should_use_autonomous_mode(user_request: str) -> tuple[bool, str]:
    stripped_request = user_request.strip()
    lowered_request = stripped_request.lower()

    if len(stripped_request) <= 30 and any(word in lowered_request for word in CHITCHAT_WORDS):
        return False, "输入更像寒暄或自我介绍，不属于需要任务队列推进的目标。"

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
    completed_ids = {task.id for task in state.tasks if task.status == "completed"}
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

请只完成当前任务，不要假装已经完成整个目标。"""


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
    tool_agent_runner: Callable[..., dict[str, Any]] = agent_runtime.run_agent_pro,
) -> dict[str, Any]:
    prompt = build_task_prompt(task, state)
    result = tool_agent_runner(
        prompt,
        use_web=state.goal.constraints.get("use_web", True),
        top_k=state.goal.constraints.get("top_k", 3),
        web_max_results=state.goal.constraints.get("web_max_results", 2),
        preferred_sources=preferred_sources,
    )
    return {
        "success": bool(result.get("answer")),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "steps": result.get("steps", []),
        "error": result.get("error", ""),
    }


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
    )


def check_stop_conditions(state: AutonomousState) -> dict[str, Any]:
    max_steps = state.goal.constraints.get("max_steps", 5)
    if all(task.status == "completed" for task in state.tasks):
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
    final_artifact = state.artifacts.get("final_deliverable")
    if final_artifact:
        return str(final_artifact)

    artifact_lines = []
    for key, value in state.artifacts.items():
        artifact_lines.append(f"## {key}\n{value}")

    if artifact_lines:
        return "\n\n".join([
            f"当前自主任务已停止，原因：{state.stop_reason}。",
            *artifact_lines,
        ])

    return f"当前自主任务未产出可用结果，停止原因：{state.stop_reason}。"


def run_autonomous_agent(
    user_request: str,
    top_k: int = 3,
    web_max_results: int = 2,
    max_steps: int = 3,
    preferred_sources: list[str] | None = None,
    tool_agent_runner: Callable[..., dict[str, Any]] = agent_runtime.run_agent_pro,
) -> dict[str, Any]:
    preferred_sources = preferred_sources or []
    goal = create_goal(user_request, max_steps=max_steps, top_k=top_k, web_max_results=web_max_results)
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
