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
AUDIT_PATH = MEMORY_PATH.with_suffix(".audit.jsonl")

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
MEMORY_STATUS_EXPIRED = "expired"
MEMORY_STATUS_SUPERSEDED = "superseded"

RISK_LEVELS = {"low", "medium", "high", "blocked"}
WRITE_DECISION_AUTO = "auto_write"
WRITE_DECISION_CONFIRM = "ask_user_confirm"
WRITE_DECISION_BLOCK = "block"

SENSITIVE_PATTERNS = [
    (r"sk-[A-Za-z0-9_\-]{12,}", "api_key"),
    (r"(?i)(api[_-]?key|secret|token|password|密码|口令)\s*[:=：]\s*\S{6,}", "secret"),
    (r"\b\d{15}(\d{2}[0-9Xx])?\b", "id_number"),
    (r"\b\d{13,19}\b", "bank_or_card_number"),
]

MEDIUM_RISK_WORDS = ["公司", "雇主", "岗位", "收入", "商业项目", "客户", "职业目标"]
HIGH_RISK_WORDS = ["住址", "地址", "健康", "医疗", "病历", "身份证", "银行卡", "薪资"]

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


def write_audit(event: str, memory_id: str = "", payload: dict | None = None) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "memory_id": memory_id,
        "ts": now_ts(),
        "payload": payload or {},
    }
    with open(AUDIT_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_audit(limit: int = 50) -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    rows = []
    for line in AUDIT_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


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
    value = item.get("value", "").strip()
    risk_level = item.get("risk_level") or classify_risk(value)
    status = item.get("status", MEMORY_STATUS_ACTIVE)
    return {
        "id": item.get("id") or new_memory_id(),
        "type": memory_type,
        "key": item.get("key", "note").strip() or "note",
        "value": value,
        "scope": item.get("scope", "global").strip() or "global",
        "status": status,
        "confidence": float(item.get("confidence", 0.8)),
        "source": item.get("source", "manual"),
        "risk_level": risk_level,
        "tags": item.get("tags", []),
        "created_at": int(item.get("created_at", timestamp)),
        "updated_at": int(item.get("updated_at", timestamp)),
        "expires_at": int(item.get("expires_at", 0) or 0),
        "last_used_at": int(item.get("last_used_at", 0)),
        "use_count": int(item.get("use_count", 0)),
        "supersedes": item.get("supersedes", []),
        "deleted_reason": item.get("deleted_reason", ""),
        "quality_score": float(item.get("quality_score", compute_quality_score({
            "confidence": item.get("confidence", 0.8),
            "risk_level": risk_level,
            "source": item.get("source", "manual"),
            "use_count": item.get("use_count", 0),
        }))),
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
    if item.get("risk_level") not in RISK_LEVELS:
        errors.append("risk_level 必须是 low、medium、high 或 blocked。")
    if item.get("risk_level") == "blocked":
        errors.append("该内容命中禁止记忆规则，不能写入 Memory。")
    return errors


def classify_risk(value: str) -> str:
    for pattern, _reason in SENSITIVE_PATTERNS:
        if re.search(pattern, value):
            return "blocked"
    if any(word in value for word in HIGH_RISK_WORDS):
        return "high"
    if any(word in value for word in MEDIUM_RISK_WORDS):
        return "medium"
    return "low"


def memory_write_decision(item: dict) -> str:
    risk = item.get("risk_level") or classify_risk(item.get("value", ""))
    if risk == "blocked":
        return WRITE_DECISION_BLOCK
    if risk in {"medium", "high"}:
        return WRITE_DECISION_CONFIRM
    return WRITE_DECISION_AUTO


def compute_quality_score(item: dict) -> float:
    source = item.get("source", "")
    source_score = 1.0 if source in {"user_confirmed", "explicit_remember", "teaching_session_summary"} else 0.75
    confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))
    usage = min(1.0, int(item.get("use_count", 0)) / 10)
    risk_penalty = {"low": 0.0, "medium": 0.08, "high": 0.2, "blocked": 1.0}.get(item.get("risk_level", "low"), 0.0)
    return round(max(0.0, source_score * 0.35 + confidence * 0.45 + usage * 0.2 - risk_penalty), 3)


def is_duplicate_or_subset(old_value: str, new_value: str) -> bool:
    old_compact = re.sub(r"\s+", "", old_value)
    new_compact = re.sub(r"\s+", "", new_value)
    if not old_compact or not new_compact:
        return False
    if old_compact in new_compact or new_compact in old_compact:
        return True
    old_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", old_value.lower()))
    new_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", new_value.lower()))
    if not old_terms or not new_terms:
        return False
    overlap = len(old_terms & new_terms) / max(1, len(old_terms | new_terms))
    return overlap >= 0.82


def merge_values(old_value: str, new_value: str) -> str:
    if re.sub(r"\s+", "", old_value) in re.sub(r"\s+", "", new_value):
        return new_value
    if re.sub(r"\s+", "", new_value) in re.sub(r"\s+", "", old_value):
        return old_value
    return f"{old_value}；补充：{new_value}"


def upsert_memory(item: dict) -> dict:
    normalized = normalize_item(item)
    errors = validate_memory_item(normalized)
    if errors:
        write_audit("memory_blocked" if normalized.get("risk_level") == "blocked" else "memory_validation_failed", normalized.get("id", ""), {
            "errors": errors,
            "type": normalized.get("type"),
            "key": normalized.get("key"),
            "risk_level": normalized.get("risk_level"),
        })
        return {"ok": False, "errors": errors, "item": normalized}

    memories = load_memories(include_deleted=True)
    for old in memories:
        same_slot = (
            old.get("type") == normalized["type"]
            and old.get("key") == normalized["key"]
            and old.get("scope", "global") == normalized["scope"]
            and old.get("status") == MEMORY_STATUS_ACTIVE
            and old.get("id") != normalized["id"]
        )
        if same_slot and normalized["type"] in {"user_profile", "user_preference", "task_progress", "semantic_rule"}:
            if is_duplicate_or_subset(old.get("value", ""), normalized["value"]):
                normalized["value"] = merge_values(old.get("value", ""), normalized["value"])
                normalized["created_at"] = old.get("created_at", normalized["created_at"])
                normalized["use_count"] = max(int(old.get("use_count", 0)), normalized["use_count"])
                normalized["last_used_at"] = max(int(old.get("last_used_at", 0)), normalized["last_used_at"])
                old["status"] = MEMORY_STATUS_SUPERSEDED
                old["deleted_reason"] = "merged_into_new_memory"
            else:
                old["status"] = MEMORY_STATUS_SUPERSEDED
                old["deleted_reason"] = "replaced_by_new_memory"
            old["updated_at"] = now_ts()
            normalized["supersedes"] = sorted(set(normalized.get("supersedes", []) + [old.get("id")]))

    replaced = False
    for index, old in enumerate(memories):
        if old.get("id") == normalized["id"]:
            normalized["created_at"] = old.get("created_at", normalized["created_at"])
            normalized["quality_score"] = compute_quality_score(normalized)
            memories[index] = normalized
            replaced = True
            break
    if not replaced:
        normalized["quality_score"] = compute_quality_score(normalized)
        memories.append(normalized)
    save_memories(memories)
    write_audit("memory_updated" if replaced else "memory_created", normalized["id"], {
        "type": normalized["type"],
        "key": normalized["key"],
        "scope": normalized["scope"],
        "risk_level": normalized["risk_level"],
        "supersedes": normalized.get("supersedes", []),
    })
    return {"ok": True, "errors": [], "item": normalized}


def update_memory_status(memory_id: str, status: str, reason: str = "") -> bool:
    memories = load_memories(include_deleted=True)
    changed = False
    for item in memories:
        if item.get("id") == memory_id:
            item["status"] = status
            if reason:
                item["deleted_reason"] = reason
            item["updated_at"] = now_ts()
            changed = True
            break
    if changed:
        save_memories(memories)
        write_audit("memory_status_changed", memory_id, {"status": status, "reason": reason})
    return changed


def delete_memory(memory_id: str, reason: str = "user_request_no_longer_use") -> bool:
    return update_memory_status(memory_id, MEMORY_STATUS_DELETED, reason)


def hard_delete_memory(memory_id: str, reason: str = "privacy_or_security_delete") -> bool:
    memories = load_memories(include_deleted=True)
    kept = [item for item in memories if item.get("id") != memory_id]
    if len(kept) == len(memories):
        return False
    save_memories(kept)
    write_audit("memory_hard_deleted", memory_id, {"reason": reason})
    return True


def restore_memory(memory_id: str) -> bool:
    return update_memory_status(memory_id, MEMORY_STATUS_ACTIVE)


def expire_due_memories(now: int | None = None) -> int:
    now = now or now_ts()
    memories = load_memories(include_deleted=True)
    changed = 0
    for item in memories:
        expires_at = int(item.get("expires_at", 0) or 0)
        if item.get("status") == MEMORY_STATUS_ACTIVE and expires_at and expires_at <= now:
            item["status"] = MEMORY_STATUS_EXPIRED
            item["updated_at"] = now
            item["deleted_reason"] = "expired"
            changed += 1
    if changed:
        save_memories(memories)
        write_audit("memory_expired", payload={"count": changed})
    return changed


def run_maintenance() -> dict:
    expired = expire_due_memories()
    memories = load_memories(include_deleted=True)
    touched = 0
    active_by_slot: dict[tuple[str, str, str], list[dict]] = {}
    for item in memories:
        if item.get("status") == MEMORY_STATUS_ACTIVE:
            slot = (item.get("type", ""), item.get("key", ""), item.get("scope", "global"))
            active_by_slot.setdefault(slot, []).append(item)

    for items in active_by_slot.values():
        if len(items) <= 1:
            continue
        items.sort(key=lambda item: (item.get("confidence", 0), item.get("updated_at", 0)), reverse=True)
        keeper = items[0]
        for duplicate in items[1:]:
            if is_duplicate_or_subset(keeper.get("value", ""), duplicate.get("value", "")):
                keeper["value"] = merge_values(keeper.get("value", ""), duplicate.get("value", ""))
                keeper["supersedes"] = sorted(set(keeper.get("supersedes", []) + [duplicate.get("id")]))
                keeper["updated_at"] = now_ts()
                duplicate["status"] = MEMORY_STATUS_SUPERSEDED
                duplicate["deleted_reason"] = "maintenance_duplicate_merge"
                duplicate["updated_at"] = now_ts()
                touched += 1

    for item in memories:
        item["quality_score"] = compute_quality_score(item)

    if touched:
        save_memories(memories)
        write_audit("memory_maintenance_merge", payload={"merged": touched})
    elif memories:
        save_memories(memories)
    return {"expired": expired, "merged": touched}


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
        and item.get("risk_level") != "blocked"
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
            item["quality_score"] = compute_quality_score(item)
            changed = True
    if changed:
        save_memories(memories)
        for memory_id in ids:
            write_audit("memory_used", memory_id)


def build_memory_context(memories: list[dict]) -> str:
    if not memories:
        return ""
    grouped: dict[str, list[str]] = {}
    for item in memories:
        grouped.setdefault(item["type"], []).append(item["value"])

    lines = ["【长期记忆】以下信息来自 Memory（记忆）系统，用于保持个性化和任务连续性；它不是权威事实证据。"]
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
        "scope": "global",
        "confidence": 0.9 if source == "explicit_remember" else 0.78,
        "source": source,
        "risk_level": classify_risk(value),
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
        "expired": sum(1 for item in memories if item.get("status") == MEMORY_STATUS_EXPIRED),
        "superseded": sum(1 for item in memories if item.get("status") == MEMORY_STATUS_SUPERSEDED),
    }


def candidate_id(candidate: dict) -> str:
    raw = f"{candidate.get('type')}|{candidate.get('key')}|{candidate.get('value')}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"cand_{digest}"
