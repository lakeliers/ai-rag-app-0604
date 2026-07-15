"""Regression coverage for session-first personal memory lookups."""

import unittest
from unittest.mock import patch

from agent_runtime import IntentResult, load_memory_after_intent, tool_direct_answer


class SessionMemoryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.context = "用户：你好，我是萧玄\n助手：你好，萧玄！"
        self.intent = IntentResult(
            intent="chitchat",
            confidence=0.95,
            reason="问候",
        )

    @patch("agent_runtime.memory_manager.retrieve_memories")
    def test_name_lookup_prefers_session_memory_before_long_term(self, retrieve_memories):
        context, memories, route = load_memory_after_intent(
            question="你记得我叫什么吗？",
            intent=self.intent,
            enabled=True,
            route_strategy="auto",
            conversation_context=self.context,
        )

        self.assertEqual(context, "")
        self.assertEqual(memories, [])
        self.assertEqual(route["source"], "session_memory")
        self.assertFalse(route["need_memory"])
        retrieve_memories.assert_not_called()

    def test_direct_answer_reads_explicit_session_name_without_model_call(self):
        result = tool_direct_answer(
            question="你记得我叫什么吗？",
            conversation_context=self.context,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.data, "你刚才说你是萧玄。")


if __name__ == "__main__":
    unittest.main()
