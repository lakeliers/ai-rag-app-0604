"""Regression coverage for the DashScope embedding integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rag_agent_core


class CloudEmbeddingTests(unittest.TestCase):
    def setUp(self):
        rag_agent_core._embedding_client = None

    def tearDown(self):
        rag_agent_core._embedding_client = None

    def test_cloud_embeddings_are_batched_and_kept_in_input_order(self):
        client = MagicMock()

        def create(**kwargs):
            rows = [
                SimpleNamespace(index=index, embedding=[float(index), 1.0, 2.0])
                for index, _ in enumerate(kwargs["input"])
            ]
            return SimpleNamespace(data=list(reversed(rows)))

        client.embeddings.create.side_effect = create
        with (
            patch.object(rag_agent_core, "EMBEDDING_PROVIDER", "dashscope"),
            patch.object(rag_agent_core, "EMBEDDING_DIMENSIONS", 3),
            patch.object(rag_agent_core, "EMBEDDING_BATCH_SIZE", 2),
            patch("rag_agent_core.get_embedding_client", return_value=client),
        ):
            vectors = rag_agent_core.embed_texts(["a", "b", "c"])

        self.assertEqual(len(vectors), 3)
        self.assertEqual(vectors[0], [0.0, 1.0, 2.0])
        self.assertEqual(client.embeddings.create.call_count, 2)

    def test_cloud_embedding_rejects_unexpected_dimensions(self):
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])]
        )
        with (
            patch.object(rag_agent_core, "EMBEDDING_PROVIDER", "dashscope"),
            patch.object(rag_agent_core, "EMBEDDING_DIMENSIONS", 3),
            patch("rag_agent_core.get_embedding_client", return_value=client),
        ):
            with self.assertRaises(rag_agent_core.EmbeddingServiceError):
                rag_agent_core.embed_texts(["dimension mismatch"])

    def test_vector_failure_is_exposed_in_diagnostics(self):
        diagnostics = {}
        with patch("rag_agent_core.embed_texts", side_effect=RuntimeError("timeout")):
            rows = rag_agent_core.vector_retrieve("test", diagnostics=diagnostics)

        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["status"], "degraded")
        self.assertIn("timeout", diagnostics["error"])

    def test_cloud_collection_is_isolated_from_legacy_vectors(self):
        self.assertNotEqual(rag_agent_core.COLLECTION_NAME, rag_agent_core.LEGACY_COLLECTION_NAME)
        self.assertIn("1024", rag_agent_core.COLLECTION_NAME)


if __name__ == "__main__":
    unittest.main()
