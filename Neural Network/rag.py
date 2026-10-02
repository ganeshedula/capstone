"""Optional lightweight Wikipedia evidence retrieval."""
from __future__ import annotations
import json
import urllib.parse
import urllib.request


def retrieve_evidence(question, limit=3, timeout=8):
    """Return short Wikipedia snippets; empty list is a retrieval failure/no-result."""
    query = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": question,
                                    "gsrlimit": limit, "prop": "extracts", "exintro": 1,
                                    "explaintext": 1, "format": "json"})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + query,
                                 headers={"User-Agent": "CapstoneHallucinationResearch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return [], f"retrieval unavailable: {exc}"
    pages = payload.get("query", {}).get("pages", {})
    ordered = sorted(pages.values(), key=lambda item: item.get("index", 10**9))
    return [p.get("extract", "").strip()[:1600] for p in ordered if p.get("extract")], None
