"""WikiRag-style evidence retrieval backed by Qdrant and DuckDuckGo."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _settings() -> dict[str, Any]:
    """Load RAG defaults from config.yaml, with environment overrides."""
    config_path = ROOT_DIR / "config.yaml"
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except Exception:
        config = {}
    configured = config.get("rag", {}) if isinstance(config, dict) else {}
    if not isinstance(configured, dict):
        configured = {}
    try:
        min_relevance = float(os.getenv("WIKIRAG_MIN_RELEVANCE", configured.get("min_relevance", 0.35)))
    except (TypeError, ValueError):
        min_relevance = 0.35
    return {
        "qdrant_url": os.getenv("WIKIRAG_QDRANT_URL", configured.get("qdrant_url", "http://localhost:6333")),
        "qdrant_api_key": os.getenv("WIKIRAG_QDRANT_API_KEY", ""),
        "collection": os.getenv("WIKIRAG_QDRANT_COLLECTION", configured.get("collection", "wikipedia")),
        "embedding_model": os.getenv("WIKIRAG_EMBEDDING_MODEL", configured.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
        "min_relevance": min_relevance,
        "expand_context": os.getenv("WIKIRAG_EXPAND_CONTEXT", str(configured.get("expand_context", True))).lower() in {"1", "true", "yes", "on"},
    }


@lru_cache(maxsize=4)
def _embeddings(embedding_model: str):
    """Load WikiRag's Hugging Face embedding adapter once per model name."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=embedding_model)


@lru_cache(maxsize=4)
def _qdrant_client(qdrant_url: str, api_key: str, timeout: int):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=qdrant_url,
        api_key=api_key or None,
        timeout=timeout,
        check_compatibility=False,
    )


def _retrieve_from_qdrant(question: str, limit: int, settings: dict, timeout: int):
    client = _qdrant_client(settings["qdrant_url"], settings.get("qdrant_api_key", ""), timeout)
    if not client.collection_exists(settings["collection"]):
        return []
    vector = _embeddings(settings["embedding_model"]).embed_query(question)
    response = client.query_points(
        collection_name=settings["collection"],
        query=vector,
        limit=limit,
        with_payload=True,
    )
    matches = []
    for point in response.points:
        payload = dict(point.payload or {})
        content = payload.get("content") or payload.get("page_content") or payload.get("text") or ""
        metadata = dict(payload.get("metadata") or {})
        metadata.update({key: value for key, value in payload.items() if key not in {"content", "page_content", "text", "metadata"}})
        document = type("RetrievedDocument", (), {"page_content": content, "metadata": metadata})()
        matches.append((document, point.score))
    return matches


def _search_web(question: str, limit: int, timeout: int) -> list[dict[str, str]]:
    """Search public web results as WikiRag's context-expansion fallback."""
    from ddgs import DDGS

    with DDGS(timeout=timeout) as search:
        results = search.text(question, max_results=limit)
        return [
            {
                "text": str(item.get("body", "")).strip(),
                "source": str(item.get("title", "DuckDuckGo result")).strip(),
                "url": str(item.get("href", "")).strip(),
                "kind": "web",
            }
            for item in results
            if str(item.get("body", "")).strip()
        ]


def _fetch_wikipedia_api(question: str, limit: int = 3, timeout: int = 6) -> list[dict[str, Any]]:
    """Direct Wikipedia REST API fallback requiring no local server or credentials."""
    import json
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrsearch": question,
        "gsrlimit": max(1, limit),
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "format": "json",
    })
    req = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "CapstoneHallucinationResearch/2.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", {})
    results = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 10**9)):
        extract = str(page.get("extract", "")).strip()
        title = str(page.get("title", "")).strip()
        page_id = page.get("pageid", "")
        if extract:
            results.append({
                "text": extract[:1600],
                "source": f"Wikipedia: {title}" if title else "Wikipedia",
                "url": f"https://en.wikipedia.org/?curid={page_id}" if page_id else "https://en.wikipedia.org",
                "kind": "wikipedia",
                "score": 0.85,
            })
    return results


def _document_evidence(document, score: float | None) -> dict[str, Any]:
    metadata = dict(getattr(document, "metadata", {}) or {})
    text = str(getattr(document, "page_content", "") or "").strip()
    url = metadata.get("source") or metadata.get("url") or metadata.get("page_url") or ""
    source = metadata.get("title") or metadata.get("name") or url or "Qdrant knowledge base"
    return {"text": text, "source": str(source), "url": str(url), "kind": "knowledge_base", "score": score}


def retrieve_evidence(question, limit=3, timeout=8, expand_context=None):
    """Return WikiRag evidence records and an optional diagnostic note.

    Qdrant failures and empty collections are non-fatal. When context expansion is
    enabled, DuckDuckGo and direct Wikipedia REST search supplement weak or
    unavailable knowledge-base results.
    """
    settings = _settings()
    should_expand = settings["expand_context"] if expand_context is None else bool(expand_context)
    evidence: list[dict[str, Any]] = []
    notes: list[str] = []
    qdrant_error: Exception | None = None
    try:
        matches = _retrieve_from_qdrant(str(question), max(1, int(limit)), settings, timeout)
        for document, score in matches:
            record = _document_evidence(document, float(score) if score is not None else None)
            if record["text"]:
                evidence.append(record)
    except Exception as exc:
        qdrant_error = exc

    best_score = max((record["score"] for record in evidence if record["score"] is not None), default=0.0)
    insufficient = not evidence or best_score < settings["min_relevance"]
    if should_expand and insufficient:
        # Tier 2: Web Search via DuckDuckGo
        try:
            web_evidence = _search_web(str(question), max(1, int(limit)), timeout)
            known_urls = {item["url"] for item in evidence if item["url"]}
            evidence.extend(item for item in web_evidence if not item["url"] or item["url"] not in known_urls)
        except Exception:
            pass

        # Tier 3: Direct Wikipedia REST API fallback if still no evidence
        if not evidence:
            try:
                wiki_evidence = _fetch_wikipedia_api(str(question), max(1, int(limit)), timeout)
                known_urls = {item["url"] for item in evidence if item["url"]}
                evidence.extend(item for item in wiki_evidence if not item["url"] or item["url"] not in known_urls)
            except Exception:
                pass

    if evidence:
        return evidence[: max(1, int(limit)) * (2 if should_expand else 1)], None

    if qdrant_error is not None and not should_expand:
        notes.append(f"Qdrant retrieval unavailable: {qdrant_error}")
    notes.append("No relevant WikiRag knowledge-base or web evidence was found.")
    return [], "; ".join(notes)


def evidence_text(item: Any) -> str:
    """Format either a WikiRag record or a legacy string for prompts/verifiers."""
    if isinstance(item, dict):
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "")).strip()
        url = str(item.get("url", "")).strip()
        attribution = " — ".join(value for value in (source, url) if value)
        return f"{text} [Source: {attribution}]" if text and attribution else text
    return str(item).strip()
