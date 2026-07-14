import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


RUN_STORE_PATH = Path(os.getenv("RUN_STORE_PATH", "data/run_lifecycle.json"))
RUN_LOCK_PATH = RUN_STORE_PATH.with_suffix(RUN_STORE_PATH.suffix + ".lock")

STATUS_CREATED = "created"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_WAITING_USER = "waiting_user"
STATUS_RETRYING = "retrying"
STATUS_CANCEL_REQUESTED = "cancel_requested"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_EXPIRED = "expired"

TERMINAL_STATUSES = {
    STATUS_SUCCEEDED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    STATUS_EXPIRED,
}

ALLOWED_TRANSITIONS = {
    STATUS_CREATED: {STATUS_QUEUED, STATUS_RUNNING, STATUS_CANCEL_REQUESTED, STATUS_FAILED},
    STATUS_QUEUED: {STATUS_RUNNING, STATUS_CANCEL_REQUESTED, STATUS_FAILED},
    STATUS_RUNNING: {
        STATUS_WAITING_USER,
        STATUS_RETRYING,
        STATUS_CANCEL_REQUESTED,
        STATUS_SUCCEEDED,
        STATUS_FAILED,
    },
    STATUS_WAITING_USER: {STATUS_QUEUED, STATUS_RUNNING, STATUS_CANCEL_REQUESTED, STATUS_EXPIRED},
    STATUS_RETRYING: {STATUS_QUEUED, STATUS_RUNNING, STATUS_CANCEL_REQUESTED, STATUS_FAILED},
    STATUS_CANCEL_REQUESTED: {STATUS_CANCELLED},
    STATUS_SUCCEEDED: set(),
    STATUS_FAILED: set(),
    STATUS_CANCELLED: set(),
    STATUS_EXPIRED: set(),
}

STATUS_LABELS = {
    STATUS_CREATED: "已创建",
    STATUS_QUEUED: "排队中",
    STATUS_RUNNING: "执行中",
    STATUS_WAITING_USER: "待用户补充",
    STATUS_RETRYING: "重试中",
    STATUS_CANCEL_REQUESTED: "正在取消",
    STATUS_SUCCEEDED: "已成功",
    STATUS_FAILED: "已失败",
    STATUS_CANCELLED: "已取消",
    STATUS_EXPIRED: "已过期",
}

_PROCESS_LOCK = threading.RLock()


def now_ts() -> int:
    return int(time.time())


def generate_request_id() -> str:
    return f"req_{uuid4().hex[:12]}"


def generate_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"


def generate_trace_id() -> str:
    return f"trace_{uuid4().hex[:12]}"


@contextmanager
def _store_lock(store_path: Path):
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass


def _load_store_unlocked(store_path: Path) -> dict[str, dict]:
    if not store_path.exists():
        return {}
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = store_path.with_suffix(store_path.suffix + f".corrupt_{stamp}")
        try:
            store_path.replace(backup)
        except OSError:
            pass
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_unlocked(store_path: Path, data: dict[str, dict]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{store_path.name}.",
        suffix=".tmp",
        dir=str(store_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, store_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _mutate_run(run_id: str, mutator, *, store_path: Path = RUN_STORE_PATH) -> dict:
    store_path = Path(store_path)
    with _store_lock(store_path):
        store = _load_store_unlocked(store_path)
        if run_id not in store:
            raise KeyError(f"Run 不存在：{run_id}")
        run = store[run_id]
        mutator(run)
        run["updated_at"] = now_ts()
        store[run_id] = run
        _atomic_write_unlocked(store_path, store)
        return json.loads(json.dumps(run, ensure_ascii=False))


def create_run(
    *,
    session_id: str,
    user_input: str,
    execution_mode: str = "sync",
    request_id: str = "",
    config: dict | None = None,
    store_path: Path = RUN_STORE_PATH,
) -> dict:
    request_id = request_id or generate_request_id()
    run_id = generate_run_id()
    created_at = now_ts()
    run = {
        "run_id": run_id,
        "session_id": session_id,
        "request_ids": [request_id],
        "initial_request_id": request_id,
        "user_input": user_input,
        "resume_inputs": [],
        "execution_mode": execution_mode,
        "status": STATUS_CREATED,
        "created_at": created_at,
        "updated_at": created_at,
        "current_trace_id": "",
        "trace_ids": [],
        "current_step": "request_validation",
        "step_states": {},
        "state_history": [{
            "from": "",
            "to": STATUS_CREATED,
            "actor": "backend_api",
            "reason": "后端完成请求校验并创建 Run。",
            "ts": created_at,
        }],
        "checkpoint": None,
        "pending_action": None,
        "result": None,
        "error": "",
        "config": config or {},
    }
    store_path = Path(store_path)
    with _store_lock(store_path):
        store = _load_store_unlocked(store_path)
        store[run_id] = run
        _atomic_write_unlocked(store_path, store)
    return json.loads(json.dumps(run, ensure_ascii=False))


def get_run(run_id: str, *, session_id: str = "", store_path: Path = RUN_STORE_PATH) -> dict | None:
    store_path = Path(store_path)
    with _store_lock(store_path):
        run = _load_store_unlocked(store_path).get(run_id)
    if not run or (session_id and run.get("session_id") != session_id):
        return None
    return json.loads(json.dumps(run, ensure_ascii=False))


def list_runs(*, session_id: str = "", limit: int = 20, store_path: Path = RUN_STORE_PATH) -> list[dict]:
    store_path = Path(store_path)
    with _store_lock(store_path):
        runs = list(_load_store_unlocked(store_path).values())
    if session_id:
        runs = [run for run in runs if run.get("session_id") == session_id]
    runs.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return json.loads(json.dumps(runs[:limit], ensure_ascii=False))


def transition_run(
    run_id: str,
    new_status: str,
    *,
    actor: str,
    reason: str,
    current_step: str = "",
    store_path: Path = RUN_STORE_PATH,
) -> dict:
    def mutate(run):
        old_status = run["status"]
        if new_status == old_status:
            return
        if new_status not in ALLOWED_TRANSITIONS.get(old_status, set()):
            raise ValueError(f"不允许的状态流转：{old_status} -> {new_status}")
        run["status"] = new_status
        if current_step:
            run["current_step"] = current_step
        run["state_history"].append({
            "from": old_status,
            "to": new_status,
            "actor": actor,
            "reason": reason,
            "ts": now_ts(),
        })

    return _mutate_run(run_id, mutate, store_path=store_path)


def start_attempt(run_id: str, *, trace_id: str = "", store_path: Path = RUN_STORE_PATH) -> dict:
    trace_id = trace_id or generate_trace_id()

    def mutate(run):
        run["current_trace_id"] = trace_id
        if trace_id not in run["trace_ids"]:
            run["trace_ids"].append(trace_id)
        run["current_step"] = "agent_execution"

    return _mutate_run(run_id, mutate, store_path=store_path)


def update_step(
    run_id: str,
    event: dict,
    *,
    actor: str = "backend_worker",
    store_path: Path = RUN_STORE_PATH,
) -> dict:
    step_id = str(event.get("id") or event.get("tool") or event.get("name") or "unknown_step")

    def mutate(run):
        previous = run["step_states"].get(step_id, {})
        merged = {**previous, **event, "id": step_id, "actor": actor, "updated_at": now_ts()}
        run["step_states"][step_id] = merged
        run["current_step"] = step_id

    return _mutate_run(run_id, mutate, store_path=store_path)


def wait_for_user(
    run_id: str,
    *,
    prompt: str,
    missing_fields: list[str],
    checkpoint_payload: dict | None = None,
    store_path: Path = RUN_STORE_PATH,
) -> dict:
    checkpoint_id = f"checkpoint_{uuid4().hex[:12]}"

    def save_wait_state(run):
        run["checkpoint"] = {
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
            "trace_id": run.get("current_trace_id", ""),
            "completed_steps": [
                step_id
                for step_id, step in run.get("step_states", {}).items()
                if step.get("status") in {"completed", "success", "skipped"}
            ],
            "payload": checkpoint_payload or {},
            "created_at": now_ts(),
        }
        run["pending_action"] = {
            "type": "collect_information",
            "prompt": prompt,
            "missing_fields": missing_fields,
            "created_at": now_ts(),
            "valid": True,
        }

    _mutate_run(run_id, save_wait_state, store_path=store_path)
    return transition_run(
        run_id,
        STATUS_WAITING_USER,
        actor="backend_runtime",
        reason="执行所需信息不足，已保存检查点并释放执行资源。",
        current_step="waiting_user_input",
        store_path=store_path,
    )


def resume_run(
    run_id: str,
    *,
    user_input: str,
    request_id: str = "",
    store_path: Path = RUN_STORE_PATH,
) -> dict:
    request_id = request_id or generate_request_id()
    run = get_run(run_id, store_path=store_path)
    if not run or run.get("status") != STATUS_WAITING_USER:
        raise ValueError("只有待用户补充的 Run 才能恢复。")
    if not user_input.strip():
        raise ValueError("恢复信息不能为空。")

    def mutate(item):
        item["request_ids"].append(request_id)
        item["resume_inputs"].append({"request_id": request_id, "content": user_input, "ts": now_ts()})
        if item.get("pending_action"):
            item["pending_action"]["valid"] = False
            item["pending_action"]["resolved_at"] = now_ts()

    _mutate_run(run_id, mutate, store_path=store_path)
    return transition_run(
        run_id,
        STATUS_QUEUED,
        actor="backend_api",
        reason="已接收并校验用户补充信息，Run 从检查点重新入队。",
        current_step="resume_from_checkpoint",
        store_path=store_path,
    )


def request_cancel(run_id: str, *, actor: str = "frontend_user", store_path: Path = RUN_STORE_PATH) -> dict:
    run = get_run(run_id, store_path=store_path)
    if not run:
        raise KeyError(f"Run 不存在：{run_id}")
    if run["status"] in TERMINAL_STATUSES:
        return run
    return transition_run(
        run_id,
        STATUS_CANCEL_REQUESTED,
        actor=actor,
        reason="收到用户取消请求，等待运行时安全终止。",
        current_step="cancelling",
        store_path=store_path,
    )


def complete_cancel(run_id: str, *, store_path: Path = RUN_STORE_PATH) -> dict:
    return transition_run(
        run_id,
        STATUS_CANCELLED,
        actor="backend_runtime",
        reason="执行资源已释放，任务已安全取消。",
        current_step="cancelled",
        store_path=store_path,
    )


def succeed_run(run_id: str, *, result: dict, store_path: Path = RUN_STORE_PATH) -> dict:
    def save_result(run):
        run["result"] = result
        run["error"] = ""

    _mutate_run(run_id, save_result, store_path=store_path)
    return transition_run(
        run_id,
        STATUS_SUCCEEDED,
        actor="backend_runtime",
        reason="结果校验通过并完成持久化。",
        current_step="result_persisted",
        store_path=store_path,
    )


def fail_run(run_id: str, *, error: str, store_path: Path = RUN_STORE_PATH) -> dict:
    def save_error(run):
        run["error"] = error

    _mutate_run(run_id, save_error, store_path=store_path)
    return transition_run(
        run_id,
        STATUS_FAILED,
        actor="backend_runtime",
        reason="执行失败，错误信息已保存。",
        current_step="failed",
        store_path=store_path,
    )


def combined_run_input(run: dict) -> str:
    parts = [str(run.get("user_input", "")).strip()]
    for item in run.get("resume_inputs", []):
        content = str(item.get("content", "")).strip()
        if content:
            parts.append(f"用户补充信息：{content}")
    return "\n".join(part for part in parts if part)


def detect_preflight_gate(question: str) -> dict | None:
    text = question.strip()
    travel_topic = any(word in text for word in ["旅行", "旅游", "行程", "出游", "度假"])
    planning_action = any(word in text for word in ["规划", "计划", "方案", "安排", "制定", "做一份"])
    if not (travel_topic and planning_action):
        return None

    missing = []
    if not re.search(r"(去|目的地|前往|到)[\u4e00-\u9fff]{2,10}", text):
        missing.append("目的地")
    if not re.search(r"(\d{1,2}[月.-]\d{1,2}|\d+天|周末|春节|国庆|日期|时间)", text):
        missing.append("出行时间")
    if not re.search(r"(预算|人均|每人).{0,8}\d+", text):
        missing.append("预算")
    if not re.search(r"\d+\s*(个)?\s*(人|位)", text):
        missing.append("人数")
    if not missing:
        return None
    return {
        "missing_fields": missing,
        "prompt": "为了继续完成旅行方案，请补充：" + "、".join(missing) + "。你可以直接用一句话告诉我。",
        "reason": "旅行方案缺少影响路线和预算计算的必要信息。",
    }
