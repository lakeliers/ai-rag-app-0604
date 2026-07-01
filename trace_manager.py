import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRACE_LOG_PATH = ROOT / "data" / "trace_logs.jsonl"
TRACE_ISSUE_TITLE = "[Trace Logs] RAG Agent runtime traces"
TRACE_ISSUE_LABELS = ["trace-log", "runtime-log"]


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def sanitize_for_trace(value: Any, *, max_text: int = 4000) -> Any:
    if isinstance(value, str):
        return _redact_text(value[:max_text])
    if isinstance(value, list):
        return [sanitize_for_trace(item, max_text=max_text) for item in value[:50]]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key)
            if any(secret_key in key_text.lower() for secret_key in ["key", "token", "secret", "password"]):
                sanitized[key_text] = "[REDACTED_SECRET]"
            else:
                sanitized[key_text] = sanitize_for_trace(item, max_text=max_text)
        return sanitized
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_local_trace(trace_record: dict[str, Any]) -> None:
    append_jsonl(TRACE_LOG_PATH, sanitize_for_trace(trace_record))


def find_local_trace(trace_id: str, path: Path = TRACE_LOG_PATH) -> list[dict[str, Any]]:
    if not trace_id or not path.exists():
        return []
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if trace_id not in line:
            continue
        try:
            matches.append(json.loads(line))
        except json.JSONDecodeError:
            matches.append({"raw": line})
    return matches


def github_enabled() -> bool:
    return os.getenv("ENABLE_ONLINE_TRACE_LOG", "1").strip() != "0"


def github_headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("未配置 GITHUB_TOKEN。")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_repo() -> str:
    repo = os.getenv("GITHUB_REPO", "").strip()
    if not repo or "/" not in repo:
        raise RuntimeError("未配置 GITHUB_REPO，格式应为 owner/repo。")
    return repo


def github_request(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=github_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API 失败：{exc.code} {detail}") from exc


def get_or_create_trace_issue() -> dict[str, Any]:
    repo = github_repo()
    query = f'repo:{repo} in:title "{TRACE_ISSUE_TITLE}"'
    search_url = "https://api.github.com/search/issues?q=" + urllib.parse.quote(query)
    search_result = github_request("GET", search_url)
    for item in search_result.get("items", []):
        if item.get("title") == TRACE_ISSUE_TITLE and "pull_request" not in item:
            return item

    payload = {
        "title": TRACE_ISSUE_TITLE,
        "body": (
            "线上运行 trace 统一记录。开发者可在本 Issue 评论中搜索 trace_id。\n\n"
            "每条评论是一次压缩后的运行快照，已做 key/token 脱敏。"
        ),
        "labels": TRACE_ISSUE_LABELS,
    }
    return github_request("POST", f"https://api.github.com/repos/{repo}/issues", payload)


def post_trace_comment(trace_record: dict[str, Any]) -> str:
    if not github_enabled():
        return ""
    issue = get_or_create_trace_issue()
    repo = github_repo()
    issue_number = issue.get("number")
    if not issue_number:
        raise RuntimeError("未获取到 Trace Log Issue 编号。")

    compact_record = sanitize_for_trace(trace_record, max_text=2000)
    trace_id = compact_record.get("trace_id", "")
    body = (
        f"### Trace `{trace_id}`\n\n"
        f"- created_at: {compact_record.get('created_at', '')}\n"
        f"- status: {compact_record.get('status', '')}\n"
        f"- mode: {compact_record.get('planner_mode', '')}\n"
        f"- tools: {', '.join(compact_record.get('tools_called', []) or []) or 'none'}\n"
        f"- sources: {', '.join(compact_record.get('sources_used', []) or []) or 'none'}\n\n"
        "```json\n"
        f"{json.dumps(compact_record, ensure_ascii=False, indent=2)}\n"
        "```"
    )
    result = github_request(
        "POST",
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        {"body": body},
    )
    return result.get("html_url", "")


def log_trace(trace_record: dict[str, Any], *, online: bool = True) -> dict[str, Any]:
    record = sanitize_for_trace({
        "created_at": now_iso(),
        **trace_record,
    })
    result = {
        "local_saved": False,
        "online_saved": False,
        "online_url": "",
        "error": "",
    }
    try:
        write_local_trace(record)
        result["local_saved"] = True
    except Exception as exc:
        result["error"] = f"local: {exc}"

    if online:
        try:
            online_url = post_trace_comment(record)
            result["online_url"] = online_url
            result["online_saved"] = bool(online_url)
        except Exception as exc:
            result["error"] = (result["error"] + " | " if result["error"] else "") + f"online: {exc}"

    if result["error"]:
        try:
            write_local_trace({
                "created_at": now_iso(),
                "trace_id": record.get("trace_id", ""),
                "event": "trace_log_error",
                "error": result["error"],
            })
        except Exception:
            pass
    return result


def log_badcase_link(trace_id: str, badcase_id: str, github_issue_url: str = "") -> dict[str, Any]:
    return log_trace(
        {
            "trace_id": trace_id,
            "event": "badcase_linked",
            "status": "badcase_submitted",
            "badcase_id": badcase_id,
            "badcase_issue_url": github_issue_url,
        },
        online=True,
    )
