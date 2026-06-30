import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "memory_governance_maintenance_report.html"


def esc(value: Any) -> str:
    return html.escape(str(value))


def run_in_temp_store() -> list[dict[str, Any]]:
    temp_dir = tempfile.TemporaryDirectory()
    os.environ["MEMORY_STORE_PATH"] = str(Path(temp_dir.name) / "memory_store.json")

    import importlib
    import memory_manager

    memory_manager = importlib.reload(memory_manager)
    rows: list[dict[str, Any]] = []

    def record(case_id: str, title: str, passed: bool, detail: dict[str, Any]) -> None:
        rows.append({
            "case_id": case_id,
            "title": title,
            "passed": passed,
            "detail": detail,
        })

    blocked = memory_manager.infer_memory_item("请记住我的 API key: sk-test-secret-1234567890", source="explicit_remember")
    blocked_result = memory_manager.upsert_memory(blocked)
    record(
        "memory_gov_001",
        "敏感凭证禁止写入",
        not blocked_result["ok"] and blocked_result["item"]["risk_level"] == "blocked",
        blocked_result,
    )

    medium = memory_manager.infer_memory_item("我的职业目标是成为 AI 产品经理", source="semi_auto_candidate")
    record(
        "memory_gov_002",
        "中风险记忆需要确认",
        medium["risk_level"] == "medium" and memory_manager.memory_write_decision(medium) == memory_manager.WRITE_DECISION_CONFIRM,
        {"candidate": medium, "decision": memory_manager.memory_write_decision(medium)},
    )

    first = memory_manager.upsert_memory({
        "type": "user_preference",
        "key": "teaching_style",
        "scope": "global",
        "value": "用户希望优先学习业内主流方案。",
        "confidence": 0.92,
        "source": "user_confirmed",
        "risk_level": "low",
    })
    second = memory_manager.upsert_memory({
        "type": "user_preference",
        "key": "teaching_style",
        "scope": "global",
        "value": "用户希望优先学习业内主流方案，并需要代码维度讲解。",
        "confidence": 0.95,
        "source": "user_confirmed",
        "risk_level": "low",
    })
    memories = memory_manager.load_memories(include_deleted=True)
    active_teaching = [
        item for item in memories
        if item.get("key") == "teaching_style" and item.get("status") == memory_manager.MEMORY_STATUS_ACTIVE
    ]
    superseded = [
        item for item in memories
        if item.get("key") == "teaching_style" and item.get("status") == memory_manager.MEMORY_STATUS_SUPERSEDED
    ]
    record(
        "memory_maint_001",
        "同 slot 重复/补充记忆合并并保留 superseded 旧版本",
        first["ok"] and second["ok"] and len(active_teaching) == 1 and len(superseded) == 1,
        {"active": active_teaching, "superseded": superseded},
    )

    expired = memory_manager.upsert_memory({
        "type": "task_progress",
        "key": "temporary_task",
        "scope": "session",
        "value": "用户今天临时要准备一个演示。",
        "confidence": 0.8,
        "source": "manual",
        "risk_level": "low",
        "expires_at": memory_manager.now_ts() - 1,
    })
    expired_count = memory_manager.expire_due_memories()
    expired_item = [
        item for item in memory_manager.load_memories(include_deleted=True)
        if item.get("id") == expired["item"]["id"]
    ][0]
    record(
        "memory_maint_002",
        "过期记忆转为 expired，不再默认检索",
        expired_count >= 1 and expired_item["status"] == memory_manager.MEMORY_STATUS_EXPIRED,
        {"expired_count": expired_count, "expired_item": expired_item},
    )

    soft_target = active_teaching[0]["id"]
    soft_deleted = memory_manager.delete_memory(soft_target, reason="eval_soft_delete")
    after_soft = [
        item for item in memory_manager.load_memories(include_deleted=True)
        if item.get("id") == soft_target
    ][0]
    record(
        "memory_maint_003",
        "软删除保留记录并退出 active 检索",
        soft_deleted and after_soft["status"] == memory_manager.MEMORY_STATUS_DELETED,
        {"item": after_soft, "active_ids": [item["id"] for item in memory_manager.load_memories()]},
    )

    hard = memory_manager.upsert_memory({
        "type": "user_profile",
        "key": "privacy_note",
        "scope": "global",
        "value": "用户要求删除的隐私记录占位。",
        "confidence": 0.8,
        "source": "manual",
        "risk_level": "medium",
    })
    hard_id = hard["item"]["id"]
    hard_deleted = memory_manager.hard_delete_memory(hard_id, reason="eval_privacy_delete")
    hard_exists = any(item.get("id") == hard_id for item in memory_manager.load_memories(include_deleted=True))
    record(
        "memory_gov_003",
        "硬删除移除原始 memory 内容",
        hard_deleted and not hard_exists,
        {"hard_id": hard_id, "exists_after_delete": hard_exists},
    )

    audit = memory_manager.load_audit(limit=100)
    record(
        "memory_gov_004",
        "审计日志记录写入、阻断、删除事件",
        all(event in {row.get("event") for row in audit} for event in ["memory_blocked", "memory_created", "memory_status_changed", "memory_hard_deleted"]),
        {"events": [row.get("event") for row in audit]},
    )

    temp_dir.cleanup()
    return rows


def render_report(rows: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for row in rows if row["passed"])
    case_rows = "\n".join(
        f"""
        <tr>
          <td><strong>{esc(row['case_id'])}</strong><br>{esc(row['title'])}</td>
          <td class="{'pass' if row['passed'] else 'fail'}">{'通过' if row['passed'] else '失败'}</td>
          <td><pre>{esc(json.dumps(row['detail'], ensure_ascii=False, indent=2)[:1800])}</pre></td>
        </tr>
        """
        for row in rows
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Memory Governance / Maintenance Eval</title>
  <style>
    body {{ margin: 0; background: #f6f7fb; color: #202431; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 36px 20px 64px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ color: #667085; margin-bottom: 22px; }}
    .card {{ background: #fff; border: 1px solid #e5e7ef; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    .metric {{ font-size: 28px; font-weight: 760; }}
    .pass {{ color: #047857; font-weight: 700; }}
    .fail {{ color: #dc2626; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7ef; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #e5e7ef; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #f1f5f9; }}
    pre {{ margin: 0; white-space: pre-wrap; font-size: 12px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; max-height: 260px; overflow: auto; }}
  </style>
</head>
<body>
<main>
  <h1>Memory Governance / Maintenance Eval</h1>
  <div class="sub">验证 memory maintenance 与 governance 的存储层能力：敏感阻断、确认策略、合并、过期、软删除、硬删除、审计日志。</div>
  <div class="card"><div class="metric">{passed}/{len(rows)}</div><div>通过样本</div></div>
  <table>
    <thead><tr><th>Case</th><th>结果</th><th>细节</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
</main>
</body>
</html>"""
    REPORT_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    rows = run_in_temp_store()
    render_report(rows)
    passed = sum(1 for row in rows if row["passed"])
    print(f"Total: {len(rows)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(rows) - passed}")
    print(f"Report: {REPORT_PATH.resolve()}")
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
