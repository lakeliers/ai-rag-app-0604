import unittest
from unittest.mock import patch

import autonomous_agent


class AutonomousRepairRegressionTest(unittest.TestCase):
    def test_first_failed_collection_gets_one_repair_and_can_finish(self):
        calls = {"count": 0}

        def fake_tool_agent_runner(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return {
                    "answer": "第一次收集没有拿到可引用资料。",
                    "sources": [],
                    "steps": [],
                }
            return {
                "answer": "补救收集成功，并获得了可引用资料。",
                "sources": [{"source": "fixture", "document": "有效资料正文"}],
                "steps": [],
            }

        def fake_synthesis(task, state, prompt, model_name=""):
            return {
                "success": True,
                "answer": ("这是满足当前阶段要求的结构化产物。" * 12),
                "sources": state.sources,
                "steps": [],
                "error": "",
            }

        with patch.object(autonomous_agent, "execute_synthesis_task", fake_synthesis):
            result = autonomous_agent.run_autonomous_agent(
                "调研三个产品并输出结构化对比报告",
                max_steps=3,
                tool_agent_runner=fake_tool_agent_runner,
            )

        tasks = {task.id: task for task in result["tasks"]}
        self.assertEqual(calls["count"], 2)
        self.assertIn("repair_collect_context", tasks)
        self.assertEqual(tasks["collect_context"].status, "repaired")
        self.assertEqual(result["stop_reason"], "all_tasks_completed")
        self.assertNotIn("no_ready_task", result["answer"])


if __name__ == "__main__":
    unittest.main()
