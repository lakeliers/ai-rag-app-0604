"""Regression coverage for non-blocking vector retrieval fallback."""

import unittest
from unittest.mock import patch

import rag_agent_core


class RagVectorFallbackTests(unittest.TestCase):
    def setUp(self):
        rag_agent_core._embedding_model = None
        rag_agent_core.embedding_load_failed = False

    def tearDown(self):
        rag_agent_core._embedding_model = None
        rag_agent_core.embedding_load_failed = False

    @patch("rag_agent_core.SentenceTransformer", side_effect=RuntimeError("model is not cached"))
    @patch("rag_agent_core.get_collection")
    def test_vector_retrieve_returns_quickly_when_model_is_unavailable(self, get_collection, _model):
        with patch.object(rag_agent_core, "EMBEDDING_PROVIDER", "local"):
            results = rag_agent_core.vector_retrieve("RAG 是什么？")

        self.assertEqual(results, [])
        get_collection.assert_not_called()

    @patch("rag_agent_core.vector_retrieve", return_value=[])
    @patch("rag_agent_core.bm25_retrieve")
    def test_search_keeps_bm25_results_when_vector_path_fails(self, bm25_retrieve, _vector_retrieve):
        bm25_retrieve.return_value = [
            {
                "id": "bm25-rag-definition",
                "document": "RAG 先检索资料，再让模型生成回答。",
                "source": "note.md",
                "source_type": "local",
                "chunk_index": 0,
                "chunk_type": "text",
                "metadata": {"source": "note.md"},
                "bm25_score": 1.0,
            }
        ]

        results = rag_agent_core.search_chroma("RAG 是什么？", retrieval_strategy="vector_bm25")

        self.assertEqual(len(results), 1)
        self.assertIn("RAG", results[0]["document"])


if __name__ == "__main__":
    unittest.main()
