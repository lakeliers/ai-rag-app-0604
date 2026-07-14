import tempfile
import unittest
from pathlib import Path

import run_lifecycle


class RunLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store_path = Path(self.temp_dir.name) / "runs.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_running_run(self):
        run = run_lifecycle.create_run(
            session_id="session_test",
            user_input="帮我做一份旅行方案",
            store_path=self.store_path,
        )
        return run_lifecycle.transition_run(
            run["run_id"],
            run_lifecycle.STATUS_RUNNING,
            actor="backend_api",
            reason="同步执行",
            store_path=self.store_path,
        )

    def test_create_and_complete_sync_run(self):
        run = self.create_running_run()
        run = run_lifecycle.start_attempt(run["run_id"], store_path=self.store_path)
        run = run_lifecycle.update_step(
            run["run_id"],
            {"id": "intent", "status": "completed", "summary": "完成意图识别"},
            store_path=self.store_path,
        )
        run = run_lifecycle.succeed_run(
            run["run_id"],
            result={"answer_preview": "方案已完成"},
            store_path=self.store_path,
        )

        self.assertEqual(run["status"], run_lifecycle.STATUS_SUCCEEDED)
        self.assertTrue(run["current_trace_id"].startswith("trace_"))
        self.assertEqual(run["step_states"]["intent"]["status"], "completed")

    def test_checkpoint_resume_keeps_run_id_and_changes_request_and_trace(self):
        run = self.create_running_run()
        run = run_lifecycle.start_attempt(run["run_id"], store_path=self.store_path)
        first_request_id = run["request_ids"][-1]
        first_trace_id = run["current_trace_id"]
        run = run_lifecycle.wait_for_user(
            run["run_id"],
            prompt="请补充目的地和预算",
            missing_fields=["目的地", "预算"],
            checkpoint_payload={"original_prompt": run["user_input"]},
            store_path=self.store_path,
        )

        self.assertEqual(run["status"], run_lifecycle.STATUS_WAITING_USER)
        self.assertTrue(run["checkpoint"]["checkpoint_id"].startswith("checkpoint_"))

        resumed = run_lifecycle.resume_run(
            run["run_id"],
            user_input="去成都，预算每人5000元",
            store_path=self.store_path,
        )
        resumed = run_lifecycle.transition_run(
            run["run_id"],
            run_lifecycle.STATUS_RUNNING,
            actor="backend_worker",
            reason="从断点恢复",
            store_path=self.store_path,
        )
        resumed = run_lifecycle.start_attempt(run["run_id"], store_path=self.store_path)

        self.assertEqual(resumed["run_id"], run["run_id"])
        self.assertNotEqual(resumed["request_ids"][-1], first_request_id)
        self.assertNotEqual(resumed["current_trace_id"], first_trace_id)
        self.assertIn("去成都", run_lifecycle.combined_run_input(resumed))

    def test_cancel_uses_two_phase_transition(self):
        run = self.create_running_run()
        requested = run_lifecycle.request_cancel(run["run_id"], store_path=self.store_path)
        self.assertEqual(requested["status"], run_lifecycle.STATUS_CANCEL_REQUESTED)
        cancelled = run_lifecycle.complete_cancel(run["run_id"], store_path=self.store_path)
        self.assertEqual(cancelled["status"], run_lifecycle.STATUS_CANCELLED)

    def test_invalid_transition_is_rejected(self):
        run = run_lifecycle.create_run(
            session_id="session_test",
            user_input="你好",
            store_path=self.store_path,
        )
        with self.assertRaises(ValueError):
            run_lifecycle.transition_run(
                run["run_id"],
                run_lifecycle.STATUS_SUCCEEDED,
                actor="backend_api",
                reason="非法跳过执行",
                store_path=self.store_path,
            )

    def test_session_scope_blocks_cross_session_read(self):
        run = self.create_running_run()
        self.assertIsNone(
            run_lifecycle.get_run(
                run["run_id"],
                session_id="another_session",
                store_path=self.store_path,
            )
        )

    def test_travel_preflight_only_stops_incomplete_planning_requests(self):
        gate = run_lifecycle.detect_preflight_gate("帮我做一份旅行方案")
        self.assertIsNotNone(gate)
        self.assertIn("目的地", gate["missing_fields"])
        self.assertIsNone(run_lifecycle.detect_preflight_gate("RAG 的定义是什么"))
        self.assertIsNone(
            run_lifecycle.detect_preflight_gate(
                "帮我安排去成都的旅行方案，12月30日出发，预算每人5000元，4个人"
            )
        )


if __name__ == "__main__":
    unittest.main()
