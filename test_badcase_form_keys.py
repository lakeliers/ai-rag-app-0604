import ast
from pathlib import Path
import unittest


APP_PATH = Path(__file__).with_name("app.py")


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


class BadcaseFormKeyRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))

    def test_dual_mode_does_not_render_a_second_global_badcase_form(self):
        node = function_node(self.tree, "render_dual_agent_compare_mode")
        called_names = [
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        ]
        self.assertNotIn("render_badcase_form", called_names)

    def test_badcase_form_key_is_trace_specific(self):
        node = function_node(self.tree, "render_badcase_form")
        form_calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "form"
        ]
        self.assertEqual(len(form_calls), 1)
        self.assertIsInstance(form_calls[0].args[0], ast.JoinedStr)


if __name__ == "__main__":
    unittest.main()
