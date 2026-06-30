import json
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
AUDIT_PATH = ROOT / "data" / "permission_audit.jsonl"

DECISION_ALLOW = "allow"
DECISION_REQUIRE_CONFIRMATION = "require_confirmation"
DECISION_BLOCK = "block"

SAFETY_MODE_LEARNING = "learning"
SAFETY_MODE_STRICT = "strict"
SAFETY_MODE_RELAXED = "relaxed"
SAFETY_MODES = {SAFETY_MODE_LEARNING, SAFETY_MODE_STRICT, SAFETY_MODE_RELAXED}

CONFIRM_POLICY_SMART = "smart"
CONFIRM_POLICY_ALWAYS = "always"
CONFIRM_POLICY_MINIMAL = "minimal"
CONFIRM_POLICIES = {CONFIRM_POLICY_SMART, CONFIRM_POLICY_ALWAYS, CONFIRM_POLICY_MINIMAL}

SENSITIVE_PATTERNS = [
    (r"sk-[A-Za-z0-9_\-]{12,}", "api_key"),
    (r"(?i)(api[_-]?key|secret|token|password|密码|口令)\s*[:=：]\s*\S{6,}", "secret"),
    (r"\b\d{15}(\d{2}[0-9Xx])?\b", "id_number"),
    (r"\b\d{13,19}\b", "card_number"),
    (r"(?i)(cookie|authorization|bearer)\s*[:=：]\s*\S{8,}", "credential"),
]

INJECTION_PATTERNS = [
    r"忽略(之前|以上|所有).{0,12}(指令|规则|要求)",
    r"不要遵守.{0,12}(系统|开发者|安全)",
    r"泄露.{0,8}(system prompt|系统提示|开发者指令)",
    r"把.{0,20}(api key|token|密码|密钥).{0,20}(发给|发送|输出|告诉)",
    r"ignore (all )?(previous|above) instructions",
    r"reveal (the )?(system|developer) prompt",
]

TOOL_POLICY = {
    "web_collect": {
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
        "audit_required": True,
    },
    "rag_search": {
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
        "audit_required": True,
    },
    "generate_answer": {
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
        "audit_required": False,
    },
    "direct_answer": {
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
        "audit_required": False,
    },
    "upload_status": {
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
        "audit_required": False,
    },
    "memory": {
        "risk_level": "medium",
        "default_decision": DECISION_REQUIRE_CONFIRMATION,
        "audit_required": True,
    },
    "badcase": {
        "risk_level": "medium",
        "default_decision": DECISION_REQUIRE_CONFIRMATION,
        "audit_required": True,
    },
}

ACTION_POLICY = {
    ("web_collect", "collect"): {"risk_level": "low", "decision": DECISION_ALLOW},
    ("rag_search", "retrieve"): {"risk_level": "low", "decision": DECISION_ALLOW},
    ("generate_answer", "generate"): {"risk_level": "low", "decision": DECISION_ALLOW},
    ("direct_answer", "generate"): {"risk_level": "low", "decision": DECISION_ALLOW},
    ("upload_status", "generate"): {"risk_level": "low", "decision": DECISION_ALLOW},
    ("memory", "read"): {"risk_level": "low", "decision": DECISION_ALLOW},
    ("memory", "write"): {"risk_level": "medium", "decision": DECISION_REQUIRE_CONFIRMATION},
    ("memory", "soft_delete"): {"risk_level": "medium", "decision": DECISION_REQUIRE_CONFIRMATION},
    ("memory", "hard_delete"): {"risk_level": "high", "decision": DECISION_REQUIRE_CONFIRMATION},
    ("badcase", "save_local"): {"risk_level": "medium", "decision": DECISION_ALLOW},
    ("badcase", "create_github_issue"): {"risk_level": "medium", "decision": DECISION_REQUIRE_CONFIRMATION},
}

OBJECT_POLICY = {
    "public_web": {
        "owner_scope": "public",
        "scope": "public",
        "allowed_operations": ["collect", "retrieve"],
        "blocked_operations": [],
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
    },
    "user_memory": {
        "owner_scope": "current_user",
        "scope": "private",
        "allowed_operations": ["read", "write", "soft_delete", "hard_delete"],
        "blocked_operations": ["export_all", "share_external"],
        "risk_level": "medium",
        "default_decision": DECISION_REQUIRE_CONFIRMATION,
    },
    "regression_case": {
        "owner_scope": "current_user",
        "scope": "workspace",
        "allowed_operations": ["save_local", "create_github_issue"],
        "blocked_operations": [],
        "risk_level": "medium",
        "default_decision": DECISION_REQUIRE_CONFIRMATION,
    },
    "final_answer": {
        "owner_scope": "current_session",
        "scope": "session",
        "allowed_operations": ["generate"],
        "blocked_operations": [],
        "risk_level": "low",
        "default_decision": DECISION_ALLOW,
    },
}


def now_ts() -> int:
    return int(time.time())


def new_action_id() -> str:
    return f"act_{uuid4().hex[:12]}"


def make_action(
    *,
    tool: str,
    operation: str,
    object_type: str,
    content: str = "",
    params: dict[str, Any] | None = None,
    action_id: str | None = None,
    source: str = "orchestrator",
) -> dict[str, Any]:
    return {
        "id": action_id or new_action_id(),
        "tool": tool,
        "operation": operation,
        "target": {"object_type": object_type},
        "content": content or "",
        "params": params or {},
        "source": source,
    }


def normalize_context(context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = dict(context or {})
    safety_mode = context.get("safety_mode", SAFETY_MODE_LEARNING)
    confirmation_policy = context.get("confirmation_policy", CONFIRM_POLICY_SMART)
    if safety_mode not in SAFETY_MODES:
        safety_mode = SAFETY_MODE_LEARNING
    if confirmation_policy not in CONFIRM_POLICIES:
        confirmation_policy = CONFIRM_POLICY_SMART
    context["safety_mode"] = safety_mode
    context["confirmation_policy"] = confirmation_policy
    context.setdefault("confirmed_actions", [])
    context.setdefault("prompt_injection_guard", True)
    context.setdefault("max_tool_calls", 10)
    context.setdefault("max_web_pages", 5)
    context.setdefault("tool_calls_used", 0)
    return context


def detect_patterns(text: str, patterns: list[tuple[str, str]] | list[str]) -> list[str]:
    hits = []
    for item in patterns:
        if isinstance(item, tuple):
            pattern, label = item
        else:
            pattern, label = item, item
        if re.search(pattern, text or ""):
            hits.append(label)
    return hits


def risk_rank(risk_level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(risk_level, 1)


def max_risk(*levels: str) -> str:
    return max(levels, key=risk_rank)


def apply_confirmation_policy(decision: str, risk_level: str, context: dict[str, Any]) -> str:
    mode = context["safety_mode"]
    policy = context["confirmation_policy"]
    if decision == DECISION_BLOCK:
        return decision
    if mode == SAFETY_MODE_STRICT and risk_rank(risk_level) >= 2:
        return DECISION_REQUIRE_CONFIRMATION
    if policy == CONFIRM_POLICY_ALWAYS and risk_rank(risk_level) >= 2:
        return DECISION_REQUIRE_CONFIRMATION
    if policy == CONFIRM_POLICY_MINIMAL and risk_rank(risk_level) < 3:
        return DECISION_ALLOW
    return decision


def build_confirmation_message(action: dict[str, Any], reason: str) -> str:
    operation = action.get("operation", "")
    object_type = action.get("target", {}).get("object_type", "")
    content = str(action.get("content", "")).strip()
    preview = f"：{content[:80]}" if content else ""
    return f"是否允许执行 {action.get('tool')} / {operation} 到 {object_type}{preview}？原因：{reason}"


def permission_gate(action: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = normalize_context(context)
    tool = action.get("tool", "")
    operation = action.get("operation", "")
    object_type = action.get("target", {}).get("object_type", "")
    content = str(action.get("content", ""))
    action_id = action.get("id", "")

    tool_policy = TOOL_POLICY.get(tool)
    if not tool_policy:
        return decision_result(action, DECISION_BLOCK, "high", f"未知工具 {tool}，不允许执行。", context)

    action_policy = ACTION_POLICY.get((tool, operation), {
        "risk_level": tool_policy.get("risk_level", "medium"),
        "decision": tool_policy.get("default_decision", DECISION_REQUIRE_CONFIRMATION),
    })
    object_policy = OBJECT_POLICY.get(object_type)
    if not object_policy:
        return decision_result(action, DECISION_BLOCK, "high", f"未知对象 {object_type}，不允许执行。", context)

    if operation in object_policy.get("blocked_operations", []):
        return decision_result(action, DECISION_BLOCK, "high", f"{object_type} 禁止执行 {operation}。", context)
    if operation not in object_policy.get("allowed_operations", []):
        return decision_result(action, DECISION_BLOCK, "high", f"{object_type} 未声明允许 {operation}。", context)

    sensitive_hits = detect_patterns(content, SENSITIVE_PATTERNS)
    if sensitive_hits:
        return decision_result(
            action,
            DECISION_BLOCK,
            "high",
            "动作内容包含敏感信息：" + "、".join(sorted(set(sensitive_hits))),
            context,
            signals={"sensitive_hits": sensitive_hits},
        )

    params = action.get("params", {})
    content_origin = params.get("content_origin", "")
    should_block_injection = content_origin in {"external_web", "uploaded_file", "tool_output"}
    injection_hits = detect_patterns(content, INJECTION_PATTERNS)
    if context.get("prompt_injection_guard") and injection_hits and should_block_injection:
        return decision_result(
            action,
            DECISION_BLOCK,
            "high",
            "检测到疑似 Prompt Injection 指令，外部资料不能提升为系统指令。",
            context,
            signals={"prompt_injection_hits": injection_hits},
        )

    if tool == "web_collect" and int(params.get("max_results", 0) or 0) > int(context.get("max_web_pages", 5)):
        return decision_result(action, DECISION_BLOCK, "medium", "联网读取页数超过本轮安全上限。", context)
    if int(context.get("tool_calls_used", 0) or 0) >= int(context.get("max_tool_calls", 10)):
        return decision_result(action, DECISION_BLOCK, "medium", "工具调用次数超过本轮安全上限。", context)

    risk_level = max_risk(
        tool_policy.get("risk_level", "low"),
        action_policy.get("risk_level", "low"),
        object_policy.get("risk_level", "low"),
    )
    decision = action_policy.get("decision") or object_policy.get("default_decision") or tool_policy.get("default_decision", DECISION_ALLOW)
    decision = apply_confirmation_policy(decision, risk_level, context)
    reason = "符合 Tool / Action / Object Policy。"
    if decision == DECISION_REQUIRE_CONFIRMATION and action_id in set(context.get("confirmed_actions", [])):
        decision = DECISION_ALLOW
        reason = "该动作已获得用户确认。"
    return decision_result(action, decision, risk_level, reason, context)


def decision_result(
    action: dict[str, Any],
    decision: str,
    risk_level: str,
    reason: str,
    context: dict[str, Any],
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_id": action.get("id", ""),
        "tool": action.get("tool", ""),
        "operation": action.get("operation", ""),
        "object_type": action.get("target", {}).get("object_type", ""),
        "decision": decision,
        "risk_level": risk_level,
        "reason": reason,
        "confirmation_message": build_confirmation_message(action, reason) if decision == DECISION_REQUIRE_CONFIRMATION else "",
        "safety_mode": context.get("safety_mode", SAFETY_MODE_LEARNING),
        "signals": signals or {},
    }


def write_audit(action: dict[str, Any], permission: dict[str, Any], event: str, result: dict[str, Any] | None = None) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": now_ts(),
        "event": event,
        "trace_id": action.get("params", {}).get("trace_id", ""),
        "action_id": action.get("id", ""),
        "tool": action.get("tool", ""),
        "operation": action.get("operation", ""),
        "object_type": action.get("target", {}).get("object_type", ""),
        "decision": permission.get("decision", ""),
        "risk_level": permission.get("risk_level", ""),
        "reason": permission.get("reason", ""),
        "result": result or {},
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_audit(limit: int = 50) -> list[dict[str, Any]]:
    if not AUDIT_PATH.exists():
        return []
    rows = []
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows[-limit:]
