import ast
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")


def _function_nodes(tree, function_name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]


def _contains_early_return_for_display_toggle(function_node):
    for node in ast.walk(function_node):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "plan_progress_enabled" not in test or not test.startswith("not "):
            continue
        if any(isinstance(child, ast.Return) for child in node.body):
            return True
    return False


def test_plan_events_are_recorded_when_live_display_is_disabled():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    handlers = []
    for handler in _function_nodes(tree, "handle_plan_progress"):
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
        ]
        if "merge_plan_event" in calls:
            handlers.append(handler)

    assert len(handlers) == 3
    for handler in handlers:
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
        ]
        assert "merge_plan_event" in calls
        assert not _contains_early_return_for_display_toggle(handler)


def test_completed_runs_persist_plan_independent_of_display_toggle():
    source = APP_PATH.read_text(encoding="utf-8")

    assert '"plan_steps": live_plan_steps if config["plan_progress_enabled"] else []' not in source
    assert '"plan_steps": live_plan_steps if plan_progress_enabled else []' not in source
    assert '"plan_steps": compact_steps_for_log(live_plan_steps) if config["plan_progress_enabled"] else []' not in source
    assert '"plan_steps": compact_steps_for_log(live_plan_steps) if plan_progress_enabled else []' not in source
    assert source.count('"plan_steps": live_plan_steps,') == 3
    assert source.count('"plan_steps": compact_steps_for_log(live_plan_steps),') == 3
