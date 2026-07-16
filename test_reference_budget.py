from pathlib import Path

import agent_runtime


def test_auto_budget_uses_fewer_sources_for_simple_definition():
    budget = agent_runtime.resolve_reference_budget(
        "RAG 是什么？",
        agent_runtime.REFERENCE_COUNT_AUTO,
        agent_runtime.REFERENCE_COUNT_AUTO,
    )

    assert budget["top_k"] == 3
    assert budget["web_max_results"] == 2
    assert budget["top_k_auto"] is True
    assert budget["web_max_results_auto"] is True


def test_auto_budget_can_exceed_the_old_five_item_limit():
    budget = agent_runtime.resolve_reference_budget(
        "请全面调研并横向对比五个 Agent 产品，输出完整选型报告。",
        agent_runtime.REFERENCE_COUNT_AUTO,
        agent_runtime.REFERENCE_COUNT_AUTO,
    )

    assert budget["top_k"] > 5
    assert budget["web_max_results"] > 5
    assert budget["top_k"] <= agent_runtime.AUTO_TOP_K_MAX
    assert budget["web_max_results"] <= agent_runtime.AUTO_WEB_RESULTS_MAX


def test_manual_budget_is_preserved_for_teaching_comparison():
    budget = agent_runtime.resolve_reference_budget(
        "请全面调研并输出报告。",
        14,
        11,
    )

    assert budget["top_k"] == 14
    assert budget["web_max_results"] == 11
    assert budget["top_k_auto"] is False
    assert budget["web_max_results_auto"] is False


def test_planner_choice_uses_auto_safety_cap_but_respects_manual_limit():
    assert agent_runtime.clamp_planner_reference_count(
        requested=11,
        configured=agent_runtime.REFERENCE_COUNT_AUTO,
        recommended=8,
        auto_cap=agent_runtime.AUTO_TOP_K_MAX,
    ) == 11
    assert agent_runtime.clamp_planner_reference_count(
        requested=11,
        configured=4,
        recommended=8,
        auto_cap=agent_runtime.AUTO_TOP_K_MAX,
    ) == 4


def test_rule_planner_receives_resolved_auto_budget():
    steps = agent_runtime.build_rule_based_steps(
        question="请全面调研三个竞品并输出详细方案。",
        use_web=True,
        top_k=agent_runtime.REFERENCE_COUNT_AUTO,
        web_max_results=agent_runtime.REFERENCE_COUNT_AUTO,
    )
    by_tool = {step.tool: step for step in steps}

    assert by_tool["web_collect"].args["max_results"] == 8
    assert by_tool["rag_search"].args["top_k"] == 10


def test_streamlit_exposes_auto_and_manual_reference_modes():
    source = Path("app.py").read_text(encoding="utf-8")

    assert '"引用数量策略"' in source
    assert '["Agent 自动判断", "手动设置"]' in source
    assert "REFERENCE_COUNT_AUTO" in source
    assert 'st.slider("资料条数", 1, 5' not in source
    assert 'st.slider("网页结果数", 1, 5' not in source
