import json
import os
import re
import time
import hashlib
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager


ROOT = Path(__file__).resolve().parent
DEFAULT_MEMORY_PATH = ROOT / "data" / "memory_store.json"
MEMORY_PATH = Path(os.getenv("MEMORY_STORE_PATH", str(DEFAULT_MEMORY_PATH)))
LOCK_PATH = MEMORY_PATH.with_suffix(MEMORY_PATH.suffix + ".lock")

MEMORY_TYPES = [
    "user_profile",
    "user_preference",
    "task_progress",
    "episodic_event",
    "semantic_rule",
]
MEMORY_STATUS_ACTIVE = "active"
MEMORY_STATUS_ARCHIVED = "archived"
MEMORY_STATUS_DELETED = "deleted"

TYPE_LABELS = {
    "user_profile": "User Memory（用户画像）",
    "user_preference": "User Memory（用户偏好）",
    "task_progress": "Task Memory（任务进度）",
    "episodic_event": "Episodic Memory（事件记忆）",
    "semantic_rule": "Semantic Memory（语义规律）",
}

DEFAULT_SEED_MEMORIES = [
    {
        "type": "user_profile",
        "key": "role",
        "value": "用户是正在学习 AI 产品经理知识的产品经理。",
        "confidence": 0.95,
        "source": "teaching_session_summary",
        "tags": ["learning", "ai_pm"],
    },
    {
        "type": "user_preference",
        "key": "teaching_style",
        "value": "教学时优先说明业内主流方案，不默认使用简化方案；需要代码维度的粗略讲解。",
        "confidence": 0.95,
        "source": "teaching_session_summary",
        "tags": ["teaching", "preference"],
    },
    {
        "type": "task_progress",
        "key": "learning_progress",
        "value": "已学习 RAG、Tool Agent、Autonomous Agent、Agent Eval；当前模块是 Agent Memory。",
        "confidence": 0.9,
        "source": "teaching_session_summary",
        "tags": ["progress", "learning"],
    },
]


def now_ts() -> int:
    return int(time.time())


def new_memory_id() -> str:
    return f"mem_{uuid4().hex[:12]}"


def ensure_store() -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        atomic_write_json([])


@contextmanager
def memory_file_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def backup_corrupt_store() -> None:
    if not MEMORY_PATH.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    corrupt_path = MEMORY_PATH.with_suffix(MEMORY_PATH.suffix + f".corrupt_{stamp}")
    try:
        MEMORY_PATH.replace(corrupt_path)
    except OSError:
        corrupt_path.write_text(MEMORY_PATH.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def atomic_write_json(data: list[dict]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MEMORY_PATH.with_suffix(MEMORY_PATH.suffix + f".tmp_{uuid4().hex[:8]}")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, MEMORY_PATH)


def load_memories(include_deleted: bool = False) -> list[dict]:
    ensure_store()
    with memory_file_lock():
        try:
            memories = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup_corrupt_store()
            atomic_write_json([])
            memories = []
    if include_deleted:
        return memories
    return [item for item in memories if item.get("status", MEMORY_STATUS_ACTIVE) != MEMORY_STATUS_DELETED]


def save_memories(memories: list[dict]) -> None:
    ensure_store()
    with memory_file_lock():
        atomic_write_json(memories)


def normalize_item(item: dict) -> dict:
    timestamp = now_ts()
    memory_type = item.get("type", "user_preference")
    if memory_type not in MEMORY_TYPES:
        memory_type = "user_preference"
    return {
        "id": item.get("id") or new_memory_id(),
        "type": memory_type,
        "key": item.get("key", "note").strip() or "note",
        "value": item.get("value", "").strip(),
        "status": item.get("status", MEMORY_STATUS_ACTIVE),
        "confidence": float(item.get("confidence", 0.8)),
        "source": item.get("source", "manual"),
        "risk_level": item.get("risk_level", "low"),
        "tags": item.get("tags", []),
        "created_at": int(item.get("created_at", timestamp)),
        "updated_at": int(item.get("updated_at", timestamp)),
        "last_used_at": int(item.get("last_used_at", 0)),
        "use_count": int(item.get("use_count", 0)),
    }


def seed_default_memories_if_empty() -> bool:
    memories = load_memories(include_deleted=True)
    if memories:
        return False
    save_memories([normalize_item(item) for item in DEFAULT_SEED_MEMORIES])
    return True


def validate_memory_item(item: dict) -> list[str]:
    errors = []
    if item.get("type") not in MEMORY_TYPES:
        errors.append("记忆类型无效。")
    if not item.get("key", "").strip():
        errors.append("key 不能为空。")
    if len(item.get("value", "").strip()) < 4:
        errors.append("记忆内容太短。")
    if item.get("risk_level") not in {"low", "medium", "high"}:
        errors.append("risk_level 必须是 low、medium 或 high。")
    return errors


def upsert_memory(item: dict) -> dict:
    normalized = normalize_item(item)
    errors = validate_memory_item(normalized)
    if errors:
        return {"ok": False, "errors": errors, "item": normalized}

    memories = load_memories(include_deleted=True)
    for old in memories:
        same_slot = (
            old.get("type") == normalized["type"]
            and old.get("key") == normalized["key"]
            and old.get("status") == MEMORY_STATUS_ACTIVE
            and old.get("id") != normalized["id"]
        )
        if same_slot and normalized["type"] in {"user_profile", "user_preference", "task_progress", "semantic_rule"}:
            old["status"] = MEMORY_STATUS_ARCHIVED
            old["updated_at"] = now_ts()

    replaced = False
    for index, old in enumerate(memories):
        if old.get("id") == normalized["id"]:
            normalized["created_at"] = old.get("created_at", normalized["created_at"])
            memories[index] = normalized
            replaced = True
            break
    if not replaced:
        memories.append(normalized)
    save_memories(memories)
    return {"ok": True, "errors": [], "item": normalized}


def update_memory_status(memory_id: str, status: str) -> bool:
    memories = load_memories(include_deleted=True)
    changed = False
    for item in memories:
        if item.get("id") == memory_id:
            item["status"] = status
            item["updated_at"] = now_ts()
            changed = True
            break
    if changed:
        save_memories(memories)
    return changed


def delete_memory(memory_id: str) -> bool:
    return update_memory_status(memory_id, MEMORY_STATUS_DELETED)


def restore_memory(memory_id: str) -> bool:
    return update_memory_status(memory_id, MEMORY_STATUS_ACTIVE)


def keyword_score(query: str, item: dict) -> int:
    text = f"{item.get('key', '')} {item.get('value', '')} {' '.join(item.get('tags', []))}"
    query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
    if not query_terms:
        return 0
    return sum(1 for term in query_terms if term and term in text.lower())


def retrieve_memories(query: str, limit: int = 8) -> list[dict]:
    active = [
        item for item in load_memories()
        if item.get("status", MEMORY_STATUS_ACTIVE) == MEMORY_STATUS_ACTIVE
    ]
    must_read_types = {"user_profile", "user_preference", "task_progress"}
    must_read = [item for item in active if item.get("type") in must_read_types]
    optional = [item for item in active if item.get("type") not in must_read_types]
    optional.sort(
        key=lambda item: (
            keyword_score(query, item),
            item.get("confidence", 0),
            item.get("updated_at", 0),
        ),
        reverse=True,
    )
    selected = []
    seen = set()
    for item in must_read + optional:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        selected.append(item)
        if len(selected) >= limit:
            break
    mark_used([item["id"] for item in selected])
    return selected


def mark_used(memory_ids: list[str]) -> None:
    if not memory_ids:
        return
    memories = load_memories(include_deleted=True)
    ids = set(memory_ids)
    changed = False
    for item in memories:
        if item.get("id") in ids:
            item["last_used_at"] = now_ts()
            item["use_count"] = int(item.get("use_count", 0)) + 1
            changed = True
    if changed:
        save_memories(memories)


def build_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""
    grouped: dict[str, list[str]] = {}
    for item in memories:
        grouped.setdefault(item["type"], []).append(item["value"])

    lines = ["【长期记忆】以下信息来自 Memory（记忆）系统，用于保持个性化和任务连续性；如与用户当前输入冲突，以当前输入为准。"]
    for memory_type in MEMORY_TYPES:
        values = grouped.get(memory_type, [])
        if not values:
            continue
        lines.append(f"{TYPE_LABELS.get(memory_type, memory_type)}：")
        for value in values[:4]:
            lines.append(f"- {value}")
    return "\n".join(lines)


def extract_explicit_memory(message: str) -> list[dict]:
    stripped = message.strip()
    match = re.match(r"^(请)?记住[:：,，\s]*(.+)$", stripped)
    if not match:
        return []
    value = match.group(2).strip()
    return [infer_memory_item(value, source="explicit_remember")]


def infer_memory_item(value: str, source: str = "candidate") -> dict:
    memory_type = "user_preference"
    key = "preference"
    tags = ["preference"]

    if any(word in value for word in ["我是", "我的角色", "产品经理", "学习者"]):
        memory_type = "user_profile"
        key = "role"
        tags = ["profile"]
    if any(word in value for word in ["学到", "学习进度", "当前模块", "下一步"]):
        memory_type = "task_progress"
        key = "learning_progress"
        tags = ["progress"]
    if any(word in value for word in ["规则", "原则", "以后遇到", "应该"]):
        memory_type = "semantic_rule"
        key = "working_rule"
        tags = ["rule"]
    if any(word in value for word in ["bug", "问题", "失败", "修复", "badcase"]):
        memory_type = "episodic_event"
        key = f"event_{now_ts()}"
        tags = ["event"]

    return {
        "type": memory_type,
        "key": key,
        "value": value,
        "confidence": 0.9 if source == "explicit_remember" else 0.78,
        "source": source,
        "risk_level": "low",
        "tags": tags,
    }


def suggest_memory_candidates(message: str) -> list[dict]:
    explicit = extract_explicit_memory(message)
    if explicit:
        return explicit

    stripped = message.strip()
    durable_signals = ["以后", "下次", "默认", "我希望", "我需要", "我的目标", "我的角色", "学习进度"]
    if not any(signal in stripped for signal in durable_signals):
        return []
    if len(stripped) < 8:
        return []

    return [infer_memory_item(stripped, source="semi_auto_candidate")]


def memory_stats() -> dict:
    memories = load_memories(include_deleted=True)
    active = [item for item in memories if item.get("status") == MEMORY_STATUS_ACTIVE]
    return {
        "total": len(memories),
        "active": len(active),
        "archived": sum(1 for item in memories if item.get("status") == MEMORY_STATUS_ARCHIVED),
        "deleted": sum(1 for item in memories if item.get("status") == MEMORY_STATUS_DELETED),
    }


def candidate_id(candidate: dict) -> str:
    raw = f"{candidate.get('type')}|{candidate.get('key')}|{candidate.get('value')}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"cand_{digest}"
