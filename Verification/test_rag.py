"""Dependency-light tests for the WikiRag evidence adapter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Neural Network"))

import rag


class FakeDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}


class FakePoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FakeClient:
    def collection_exists(self, collection_name):
        self.collection_name = collection_name
        return True

    def query_points(self, **kwargs):
        self.query = kwargs
        return type("QueryResponse", (), {
            "points": [FakePoint({"content": "WikiRag chunk", "title": "Wiki title", "url": "https://wiki.test/page", "language": "en"}, 0.88)]
        })()


class WikiRagAdapterTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "qdrant_url": "http://qdrant.test:6333",
            "collection": "wiki",
            "embedding_model": "test-embedding",
            "min_relevance": 0.35,
            "expand_context": True,
        }

    def test_returns_qdrant_text_and_source_metadata(self):
        document = FakeDocument("The answer is in the article.", {"title": "Sample", "source": "https://wiki.test/sample"})
        with patch.object(rag, "_settings", return_value=self.settings), \
             patch.object(rag, "_retrieve_from_qdrant", return_value=[(document, 0.91)]), \
             patch.object(rag, "_search_web") as web_search:
            evidence, note = rag.retrieve_evidence("question")

        self.assertIsNone(note)
        self.assertEqual(evidence[0]["text"], document.page_content)
        self.assertEqual(evidence[0]["source"], "Sample")
        self.assertEqual(evidence[0]["url"], "https://wiki.test/sample")
        web_search.assert_not_called()

    def test_reads_the_upstream_wikirag_qdrant_payload_shape(self):
        client = FakeClient()
        embedder = type("FakeEmbeddings", (), {"embed_query": lambda _self, _query: [0.1, 0.2]})()
        with patch.object(rag, "_embeddings", return_value=embedder), \
             patch.object(rag, "_qdrant_client", return_value=client):
            matches = rag._retrieve_from_qdrant("question", 2, self.settings, 8)

        record = rag._document_evidence(*matches[0])
        self.assertEqual(record["text"], "WikiRag chunk")
        self.assertEqual(record["source"], "Wiki title")
        self.assertEqual(record["url"], "https://wiki.test/page")
        self.assertEqual(client.collection_name, self.settings["collection"])
        self.assertEqual(client.query["limit"], 2)

    def test_skips_embedding_load_when_collection_is_absent(self):
        client = type("MissingCollectionClient", (), {"collection_exists": lambda _self, _name: False})()
        with patch.object(rag, "_qdrant_client", return_value=client), \
             patch.object(rag, "_embeddings") as embeddings:
            matches = rag._retrieve_from_qdrant("question", 2, self.settings, 8)

        self.assertEqual(matches, [])
        embeddings.assert_not_called()

    def test_expands_web_context_when_qdrant_is_empty_or_weak(self):
        web_record = {"text": "Web fact", "source": "Search result", "url": "https://web.test", "kind": "web"}
        with patch.object(rag, "_settings", return_value=self.settings), \
             patch.object(rag, "_retrieve_from_qdrant", return_value=[]), \
             patch.object(rag, "_search_web", return_value=[web_record]) as web_search:
            evidence, note = rag.retrieve_evidence("question")

        self.assertIsNone(note)
        self.assertEqual(evidence, [web_record])
        web_search.assert_called_once()

    def test_qdrant_failure_uses_web_fallback_without_warning(self):
        web_record = {"text": "Fallback fact", "source": "Search result", "url": "https://web.test", "kind": "web"}
        with patch.object(rag, "_settings", return_value=self.settings), \
             patch.object(rag, "_retrieve_from_qdrant", side_effect=ConnectionError("offline")), \
             patch.object(rag, "_search_web", return_value=[web_record]):
            evidence, note = rag.retrieve_evidence("question")

        self.assertEqual(evidence, [web_record])
        self.assertIsNone(note)

    def test_no_evidence_and_disabled_expansion_returns_clear_note(self):
        settings = {**self.settings, "expand_context": False}
        with patch.object(rag, "_settings", return_value=settings), \
             patch.object(rag, "_retrieve_from_qdrant", return_value=[]), \
             patch.object(rag, "_search_web") as web_search:
            evidence, note = rag.retrieve_evidence("question")

        self.assertEqual(evidence, [])
        self.assertIn("No relevant WikiRag", note)
        web_search.assert_not_called()

    def test_qdrant_and_web_failure_falls_back_to_wikipedia_api(self):
        wiki_record = {"text": "Wikipedia fact", "source": "Wikipedia: Sample", "url": "https://en.wikipedia.org/sample", "kind": "wikipedia", "score": 0.85}
        with patch.object(rag, "_settings", return_value=self.settings), \
             patch.object(rag, "_retrieve_from_qdrant", side_effect=ConnectionError("offline")), \
             patch.object(rag, "_search_web", side_effect=Exception("No results found.")), \
             patch.object(rag, "_fetch_wikipedia_api", return_value=[wiki_record]) as wiki_fetch:
            evidence, note = rag.retrieve_evidence("what is the capital of america?")

        self.assertIsNone(note)
        self.assertEqual(evidence, [wiki_record])
        wiki_fetch.assert_called_once()

    def test_formats_attribution_for_prompt_and_verifier(self):
        item = {"text": "A fact.", "source": "Article", "url": "https://wiki.test/article"}
        self.assertEqual(rag.evidence_text(item), "A fact. [Source: Article — https://wiki.test/article]")


if __name__ == "__main__":
    unittest.main()
