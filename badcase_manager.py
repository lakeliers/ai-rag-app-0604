import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EVAL_CASES_PATH = ROOT / "eval_cases.jsonl"
BAD_CASES_PATH = ROOT / "bad_cases.jsonl"

SUITES = ["smoke", "regression", "benchmark"]
CATEGORIES = [
    "chitchat",
    "upload_status",
    "source_scope",
    "web_rag",
    "document_qa",
    "definition",
    "autonomous",
    "autonomous_fallback",
    "hybrid_rag",
]
SEVERITIES = ["low", "medium", "high", "blocker"]
SELECTED_MODES = ["normal", "autonomous"]
EXPECTED_MODES = ["pro_runtime", "autonomous_runtime", "autonomous_fallback"]
TOOLS = [
    "direct_answer",
    "upload_status",
    "web_collect",
    "rag_search",
    "generate_answer",
    "answer_validator",
]
SOURCES = ["upload", "web", "local"]
SAVE_TARGET_LOCAL = "本地 eval 集合"
SAVE_TARGET_GITHUB = "线上 eval 集合（GitHub Issue）"
SAVE_TARGET_BOTH = "两者都保存"
SAVE_TARGETS = [SAVE_TARGET_LOCAL, SAVE_TARGET_GITHUB, SAVE_TARGET_BOTH]


def split_list(value: str) -> list[str]:
    parts = re.split(r"[,，\n]", value or "")
    return [part.strip() for part in parts if part.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def existing_case_ids(path: Path = EVAL_CASES_PATH) -> set[str]:
    return {
        str(row.get("case_id", ""))
        for row in load_jsonl(path)
        if row.get("case_id")
    }


def slugify(text: str, max_len: int = 36) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")
    return cleaned[:max_len] or "case"


def generate_case_id(user_input: str, category: str = "chitchat") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"regression_{category}_{slugify(user_input)}_{stamp}"


def generate_badcase_id() -> str:
    return f"badcase_{time.strftime('%Y%m%d_%H%M%S')}"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_automatic_check(case: dict[str, Any]) -> bool:
    check_fields = [
        "expected_mode",
        "expected_tools",
        "forbidden_tools",
        "expected_sources",
        "forbidden_sources",
        "required_phrases",
        "expected_answer_phrases",
        "forbidden_answer_phrases",
        "min_answer_chars",
        "success_criteria",
    ]
    for field in check_fields:
        value = case.get(field)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, int) and value > 0:
            return True
    return False


def validate_badcase_log(log: dict[str, Any]) -> list[str]:
    errors = []
    if not log.get("user_input", "").strip():
        errors.append("缺少 user prompt。")
    if not log.get("actual_answer", "").strip():
        errors.append("缺少 agent 真实回复。")
    if not log.get("severity"):
        errors.append("请选择严重级别。")
    if not log.get("problem_description", "").strip() and not log.get("success_criteria"):
        errors.append("请补充问题说明或成功标准，便于开发者判断。")
    return errors


def validate_regression_case(case: dict[str, Any]) -> list[str]:
    errors = []
    case_id = case.get("case_id", "").strip()
    if not case_id:
        errors.append("缺少 case_id。")
    elif not re.match(r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$", case_id):
        errors.append("case_id 只能包含中英文、数字、下划线或短横线。")
    elif case_id in existing_case_ids():
        errors.append(f"case_id 已存在：{case_id}")

    suites = case.get("suite", [])
    if not suites:
        errors.append("至少选择一个 suite。")
    elif not set(suites).issubset(SUITES):
        errors.append("suite 包含非法选项。")
    elif "regression" not in suites:
        errors.append("保存到本地 eval 集合时，suite 必须包含 regression。")

    if case.get("category") not in CATEGORIES:
        errors.append("请选择合法的 category。")
    if case.get("selected_mode") not in SELECTED_MODES:
        errors.append("请选择合法的 selected_mode。")
    if case.get("expected_mode") and case.get("expected_mode") not in EXPECTED_MODES:
        errors.append("请选择合法的 expected_mode。")

    for field in ["expected_tools", "forbidden_tools"]:
        if not set(case.get(field, [])).issubset(TOOLS):
            errors.append(f"{field} 包含非法工具。")
    for field in ["expected_sources", "forbidden_sources"]:
        if not set(case.get(field, [])).issubset(SOURCES):
            errors.append(f"{field} 包含非法来源。")

    if not case.get("user_input", "").strip():
        errors.append("缺少 user_input。")
    if not has_automatic_check(case):
        errors.append("至少需要填写一个可自动评估的约束，例如 expected_tools、forbidden_tools、required_phrases 或 forbidden_answer_phrases。")
    if len(case.get("required_phrases", [])) > 5:
        errors.append("required_phrases 不建议超过 5 个，否则容易把正确答案误判为失败。")
    if int(case.get("min_answer_chars", 0) or 0) < 0:
        errors.append("min_answer_chars 不能小于 0。")
    return errors


def build_badcase_log(
    *,
    badcase_id: str,
    case: dict[str, Any],
    actual_answer: str,
    config: dict[str, Any],
    tools_called: list[str],
    sources_used: list[str],
    severity: str,
    problem_description: str,
    note: str,
    github_issue_url: str = "",
    review_status: str = "pending_review",
) -> dict[str, Any]:
    return {
        "badcase_id": badcase_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "user_input": case.get("user_input", ""),
        "actual_answer": actual_answer,
        "severity": severity,
        "category": case.get("category", ""),
        "problem_description": problem_description,
        "suggested_regression_case": case,
        "config": config,
        "tools_called": tools_called,
        "sources_used": sources_used,
        "note": note,
        "review_status": review_status,
        "github_issue_url": github_issue_url,
    }


def format_github_issue_body(badcase_log: dict[str, Any]) -> str:
    case_json = json.dumps(
        badcase_log.get("suggested_regression_case", {}),
        ensure_ascii=False,
        indent=2,
    )
    config_json = json.dumps(badcase_log.get("config", {}), ensure_ascii=False, indent=2)
    return f"""
## User Prompt
{badcase_log.get("user_input", "")}

## Agent Answer
{badcase_log.get("actual_answer", "")}

## Problem
- Severity: {badcase_log.get("severity", "")}
- Category: {badcase_log.get("category", "")}
- Description: {badcase_log.get("problem_description", "")}

## Runtime
- Tools called: {", ".join(badcase_log.get("tools_called", [])) or "none"}
- Sources used: {", ".join(badcase_log.get("sources_used", [])) or "none"}

## Config
```json
{config_json}
```

## Suggested regression case
```json
{case_json}
```

## Note
{badcase_log.get("note", "")}
""".strip()


def create_github_issue(badcase_log: dict[str, Any]) -> str:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    if not token:
        raise RuntimeError("未配置 GITHUB_TOKEN，无法创建 GitHub Issue。")
    if not repo or "/" not in repo:
        raise RuntimeError("未配置 GITHUB_REPO，格式应为 owner/repo。")

    title_prompt = badcase_log.get("user_input", "")[:40] or badcase_log["badcase_id"]
    payload = {
        "title": f"[Bad Case] {title_prompt}",
        "body": format_github_issue_body(badcase_log),
        "labels": ["bad-case", "agent-eval", badcase_log.get("severity", "medium")],
    }
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("html_url", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub Issue 创建失败：{exc.code} {detail}") from exc


def save_case(
    *,
    save_target: str,
    case: dict[str, Any],
    actual_answer: str,
    config: dict[str, Any],
    tools_called: list[str],
    sources_used: list[str],
    severity: str,
    problem_description: str,
    note: str,
) -> dict[str, Any]:
    badcase_id = generate_badcase_id()
    result = {
        "local_badcase_saved": False,
        "local_eval_saved": False,
        "github_issue_url": "",
        "github_error": "",
        "errors": [],
    }

    local_requested = save_target in {SAVE_TARGET_LOCAL, SAVE_TARGET_BOTH}
    github_requested = save_target in {SAVE_TARGET_GITHUB, SAVE_TARGET_BOTH}

    badcase_log = build_badcase_log(
        badcase_id=badcase_id,
        case=case,
        actual_answer=actual_answer,
        config=config,
        tools_called=tools_called,
        sources_used=sources_used,
        severity=severity,
        problem_description=problem_description,
        note=note,
    )

    badcase_errors = validate_badcase_log(badcase_log)
    if badcase_errors:
        result["errors"].extend(badcase_errors)
    if local_requested:
        result["errors"].extend(validate_regression_case(case))
    if result["errors"]:
        return result

    append_jsonl(BAD_CASES_PATH, badcase_log)
    result["local_badcase_saved"] = True

    if local_requested:
        append_jsonl(EVAL_CASES_PATH, case)
        result["local_eval_saved"] = True

    if github_requested:
        try:
            issue_url = create_github_issue(badcase_log)
            result["github_issue_url"] = issue_url
            badcase_log["github_issue_url"] = issue_url
        except Exception as exc:
            result["github_error"] = str(exc)

    return result
