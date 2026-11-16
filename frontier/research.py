"""Web research tools for the frontier loop: web search, page fetch, preprint aggregation.

Keyless by default. `preprint_search` merges arXiv, OpenAlex (which indexes engrXiv, SSRN,
bioRxiv, medRxiv, ChemRxiv, TechRxiv and more), and Europe PMC preprints, de-duplicated by
title. `web_search` uses FRONTIER_SEARCH_URL when set (a SearXNG-style JSON endpoint taking
`q`), otherwise DuckDuckGo's lite endpoint.

These tools exist only in the frontier loop. The competition agent has no web access.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

USER_AGENT = "Mozilla/5.0 (compatible; frontier-loop/1.0)"
TIMEOUT = 30

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch a URL and return its readable text (HTML stripped, truncated).",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "preprint_search",
        "description": ("Search preprint servers and aggregate the results: arXiv, OpenAlex (engrXiv, "
                        "SSRN, bioRxiv, medRxiv, ChemRxiv, TechRxiv) and Europe PMC. Returns title, "
                        "year, source, URL and abstract excerpt, de-duplicated."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}}},
]


def _get(url: str, accept: str = "*/*") -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read(3_000_000).decode("utf-8", errors="replace")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _search_web(query: str, limit: int) -> list[dict[str, str]]:
    custom = os.environ.get("FRONTIER_SEARCH_URL")
    if custom:
        data = json.loads(_get(f"{custom}?{urllib.parse.urlencode({'q': query, 'format': 'json'})}"))
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in data.get("results", [])[:limit]]
    page = _get("https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query}))
    links = re.findall(r"""(?s)<a[^>]*href=["']([^"']+)["'][^>]*class=["']result-link["'][^>]*>(.*?)</a>""", page)
    snippets = re.findall(r"""(?s)<td[^>]*class=["']result-snippet["'][^>]*>(.*?)</td>""", page)
    results = []
    for index, (href, title) in enumerate(links[:limit]):
        if href.startswith("//"):
            href = "https:" + href
        target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        results.append({"title": _strip_html(title), "url": target,
                        "snippet": _strip_html(snippets[index]) if index < len(snippets) else ""})
    return results


def _arxiv(query: str, limit: int) -> list[dict[str, str]]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "max_results": limit, "sortBy": "relevance"})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    items = []
    for entry in ET.fromstring(_get(url)).findall("a:entry", ns):
        items.append({"title": " ".join((entry.findtext("a:title", "", ns)).split()),
                      "year": (entry.findtext("a:published", "", ns))[:4], "source": "arXiv",
                      "url": entry.findtext("a:id", "", ns),
                      "abstract": " ".join((entry.findtext("a:summary", "", ns)).split())[:500]})
    return items


def _openalex(query: str, limit: int) -> list[dict[str, str]]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "filter": "type:preprint", "per-page": limit})
    items = []
    for work in json.loads(_get(url)).get("results", []):
        index = work.get("abstract_inverted_index") or {}
        words = sorted((pos, word) for word, positions in index.items() for pos in positions)
        location = work.get("primary_location") or {}
        items.append({"title": work.get("title") or "", "year": str(work.get("publication_year") or ""),
                      "source": ((location.get("source") or {}).get("display_name")) or "OpenAlex",
                      "url": work.get("doi") or location.get("landing_page_url") or work.get("id", ""),
                      "abstract": " ".join(w for _, w in words)[:500]})
    return items


def _europepmc(query: str, limit: int) -> list[dict[str, str]]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": f"({query}) AND SRC:PPR", "format": "json", "pageSize": limit, "resultType": "core"})
    items = []
    for hit in json.loads(_get(url)).get("resultList", {}).get("result", []):
        doi = hit.get("doi")
        items.append({"title": hit.get("title", ""), "year": str(hit.get("pubYear", "")),
                      "source": (hit.get("bookOrReportDetails") or {}).get("publisher") or "Europe PMC preprint",
                      "url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/PPR/{hit.get('id')}",
                      "abstract": _strip_html(hit.get("abstractText", ""))[:500]})
    return items


async def web_search(args: dict[str, Any]) -> str:
    limit = min(int(args.get("max_results") or 8), 15)
    try:
        results = await asyncio.to_thread(_search_web, args["query"], limit)
    except Exception as exc:  # network errors are reported to the model, not raised
        return f"error: search failed: {type(exc).__name__}: {exc}"
    if not results:
        return "no results"
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in results)


async def web_fetch(args: dict[str, Any]) -> str:
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        return "error: only http and https URLs can be fetched"
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("localhost", "") or re.match(r"^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1)", host):
        return "error: local and private addresses are not fetched"
    try:
        body = await asyncio.to_thread(_get, url)
    except Exception as exc:
        return f"error: fetch failed: {type(exc).__name__}: {exc}"
    return _strip_html(body)[:15000] if "<" in body[:2000] else body[:15000]


async def preprint_search(args: dict[str, Any]) -> str:
    limit = min(int(args.get("max_results") or 6), 12)
    sources = {"arXiv": _arxiv, "OpenAlex": _openalex, "Europe PMC": _europepmc}
    batches = await asyncio.gather(
        *(asyncio.to_thread(fn, args["query"], limit) for fn in sources.values()), return_exceptions=True)
    seen, merged, notes = set(), [], []
    for name, batch in zip(sources, batches):
        if isinstance(batch, Exception):
            notes.append(f"({name} unavailable: {type(batch).__name__})")
            continue
        for item in batch:
            key = re.sub(r"[^a-z0-9]", "", item["title"].lower())[:80]
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    if not merged:
        return "no preprints found " + " ".join(notes)
    lines = [f"{i['title']} ({i['year']}, {i['source']})\n{i['url']}\n{i['abstract']}" for i in merged]
    return "\n\n".join(lines + notes)


HANDLERS = {"web_search": web_search, "web_fetch": web_fetch, "preprint_search": preprint_search}
