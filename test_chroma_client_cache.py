import os
import unittest
from unittest.mock import MagicMock, patch

import rag_agent_core


class ChromaClientCacheTests(unittest.TestCase):
    def setUp(self):
        rag_agent_core._client_cache.clear()
        rag_agent_core._collection_cache.clear()

    def tearDown(self):
        rag_agent_core._client_cache.clear()
        rag_agent_core._collection_cache.clear()

    @patch("rag_agent_core.chromadb.PersistentClient")
    def test_client_is_kept_alive_and_reused_for_same_path(self, persistent_client):
        client = MagicMock()
        client.get_or_create_collection.side_effect = lambda name: f"collection:{name}"
        persistent_client.return_value = client

        first = rag_agent_core.get_collection("./tmp-chroma", "documents")
        second = rag_agent_core.get_collection("./tmp-chroma", "documents")
        third = rag_agent_core.get_collection("./tmp-chroma", "memories")

        normalized_path = os.path.abspath("./tmp-chroma")
        self.assertEqual(first, "collection:documents")
        self.assertIs(first, second)
        self.assertEqual(third, "collection:memories")
        persistent_client.assert_called_once_with(path=normalized_path)
        self.assertIs(rag_agent_core._client_cache[normalized_path], client)
        self.assertEqual(client.get_or_create_collection.call_count, 2)


if __name__ == "__main__":
    unittest.main()
