import html
import json
import os
import re
import time
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote_plus
import httpx

from database import DEFAULT_USER_ID, get_provider_secret_internal
from engine.nlp_math import extract_top_keyphrases, tfidf_cosine_scores
from security import redact_sensitive, validate_external_url


class _ReadableHTMLParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside", "form"}

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self._skip_depth = 0
        self._in_title = False
        self.title = ""
        self.published_date = ""
        self.blocks: List[str] = []
        self._cur_buf: List[str] = []
        self.links: List[Dict[str, str]] = []
        self._cur_href: Optional[str] = None
        self._cur_link_text: List[str] = []

    def _flush_block(self) -> None:
        if self._cur_buf:
            txt = re.sub(r"\s+", " ", "".join(self._cur_buf)).strip()
            if len(txt) >= 3:
                self.blocks.append(txt)
            self._cur_buf.clear()

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}

        if t in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if t == "title":
            self._in_title = True
        elif t == "meta":
            prop = (attr_map.get("property") or attr_map.get("name") or "").lower()
            content = attr_map.get("content", "").strip()
            if content and not self.published_date and prop in {
                "article:published_time",
                "og:updated_time",
                "pubdate",
                "publishdate",
                "date",
                "dc.date",
                "dcterms.created",
            }:
                self.published_date = content[:25]
        elif t == "time" and not self.published_date:
            dt = attr_map.get("datetime", "").strip()
            if dt:
                self.published_date = dt[:25]
        elif t in {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "article", "section", "div", "br"}:
            self._flush_block()
        elif t == "a" and self._skip_depth == 0:
            href = attr_map.get("href", "").strip()
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self._cur_href = urljoin(self.base_url, href)
                self._cur_link_text = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if t == "title":
            self._in_title = False
        elif t in {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "article", "section", "div"}:
            self._flush_block()
        elif t == "a" and self._cur_href:
            lbl = re.sub(r"\s+", " ", "".join(self._cur_link_text)).strip()
            if lbl and len(lbl) >= 4 and self._cur_href.startswith(("http://", "https://")):
                self.links.append({"url": self._cur_href, "label": lbl[:100]})
            self._cur_href = None
            self._cur_link_text = []

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title += data.strip()
        if self._skip_depth > 0:
            return
        self._cur_buf.append(data)
        if self._cur_href is not None:
            self._cur_link_text.append(data)


def extract_readable_html(raw_html: str, url: str) -> Dict[str, Any]:
    parser = _ReadableHTMLParser(url)
    try:
        parser.feed(raw_html)
        parser._flush_block()
    except Exception:
        pass

    # Fallback JSON-LD datePublished scan
    pub_date = parser.published_date
    if not pub_date:
        m = re.search(r'"datePublished"\s*:\s*"([^"]{4,25})"', raw_html)
        if m:
            pub_date = m.group(1)

    # Deduplicate consecutive identical blocks
    dedup_blocks: List[str] = []
    for b in parser.blocks:
        clean_b = html.unescape(b).strip()
        if clean_b and (not dedup_blocks or dedup_blocks[-1] != clean_b):
            dedup_blocks.append(clean_b)

    text_body = "\n\n".join(dedup_blocks)
    seen_urls: set[str] = set()
    unique_links: List[Dict[str, str]] = []
    for lk in parser.links:
        if lk["url"] not in seen_urls and lk["url"] != url:
            seen_urls.add(lk["url"])
            unique_links.append(lk)
        if len(unique_links) >= 15:
            break

    return {
        "title": html.unescape(parser.title).strip() or urlparse(url).netloc,
        "url": url,
        "published_date": pub_date or None,
        "content": text_body[:12000],
        "links": unique_links,
    }


class WebResearchEngine:
    """
    Grounded Web Research Subsystem:
    - Supports Tavily (`TAVILY_API_KEY`), Brave Search (`BRAVE_API_KEY`), Serper (`SERPER_API_KEY`),
      DuckDuckGo HTML/Instant search, Wikipedia OpenSearch, GitHub Public Search API, and direct URL extraction.
    - Enforces SSRF protection on all outbound requests.
    - Never fabricates search results or citations; transparently reports provider status and network reachability.
    """

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "tavily_configured": bool(os.environ.get("TAVILY_API_KEY", "").strip()),
            "brave_configured": bool(os.environ.get("BRAVE_API_KEY", "").strip()),
            "serper_configured": bool(os.environ.get("SERPER_API_KEY", "").strip()),
            "public_connectors": ["duckduckgo_web", "wikipedia_api", "github_public_api", "direct_url_fetch"],
        }

    async def fetch_webpage(self, url: str, timeout_sec: float = 10.0) -> Dict[str, Any]:
        ok, reason = validate_external_url(url)
        if not ok:
            return {"ok": False, "url": url, "error": reason}

        headers = {
            "User-Agent": "XyvoraAI-ResearchBot/2.0 (+https://xyvora.net)",
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True, max_redirects=4) as client:
                resp = await client.get(url, headers=headers)
                final_url = str(resp.url)
                ok_final, reason_final = validate_external_url(final_url)
                if not ok_final:
                    return {"ok": False, "url": url, "error": reason_final}

                if resp.status_code >= 400:
                    return {"ok": False, "url": final_url, "error": f"HTTP {resp.status_code}"}

                ctype = (resp.headers.get("content-type") or "").lower()
                raw_text = resp.text[:250_000]
                if "json" in ctype:
                    return {
                        "ok": True,
                        "title": urlparse(final_url).path or final_url,
                        "url": final_url,
                        "published_date": None,
                        "content": raw_text[:10000],
                        "links": [],
                        "verified": True,
                    }
                if "html" in ctype or "<html" in raw_text[:500].lower():
                    extracted = extract_readable_html(raw_text, final_url)
                    extracted["ok"] = True
                    extracted["verified"] = True
                    return extracted

                return {
                    "ok": True,
                    "title": urlparse(final_url).netloc,
                    "url": final_url,
                    "published_date": None,
                    "content": raw_text[:10000],
                    "links": [],
                    "verified": True,
                }
        except Exception as exc:
            return {
                "ok": False,
                "url": url,
                "error": f"Could not retrieve '{url}': {redact_sensitive(str(exc))[:200]}",
            }

    async def _search_tavily(self, query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            return [], None
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": max_results, "include_answer": False},
                )
            if resp.status_code != 200:
                return [], f"Tavily API returned HTTP {resp.status_code}"
            data = resp.json()
            results = []
            for item in data.get("results") or []:
                results.append(
                    {
                        "title": item.get("title") or item.get("url", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", "")[:500],
                        "published_date": item.get("published_date"),
                        "source_provider": "tavily",
                        "verified": True,
                    }
                )
            return results, None
        except Exception as exc:
            return [], f"Tavily error: {redact_sensitive(str(exc))[:160]}"

    async def _search_brave(self, query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        api_key = os.environ.get("BRAVE_API_KEY", "").strip()
        if not api_key:
            return [], None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(query)}&count={max_results}",
                    headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                )
            if resp.status_code != 200:
                return [], f"Brave Search returned HTTP {resp.status_code}"
            web_items = ((resp.json().get("web") or {}).get("results")) or []
            results = [
                {
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "snippet": it.get("description", "")[:500],
                    "published_date": it.get("page_age") or it.get("age"),
                    "source_provider": "brave",
                    "verified": True,
                }
                for it in web_items[:max_results]
            ]
            return results, None
        except Exception as exc:
            return [], f"Brave Search error: {redact_sensitive(str(exc))[:160]}"

    async def _search_duckduckgo(self, query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(
                    f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                    headers={"User-Agent": "Mozilla/5.0 (compatible; XyvoraResearch/2.0)"},
                )
            if resp.status_code != 200:
                return [], f"DuckDuckGo returned HTTP {resp.status_code}"
            body = resp.text
            matches = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                body,
                re.DOTALL,
            )
            results: List[Dict[str, Any]] = []
            for href, raw_t, raw_s in matches[:max_results]:
                clean_title = html.unescape(re.sub(r"<[^>]+>", "", raw_t)).strip()
                clean_snip = html.unescape(re.sub(r"<[^>]+>", "", raw_s)).strip()
                if href.startswith("//"):
                    href = "https:" + href
                results.append(
                    {
                        "title": clean_title,
                        "url": href,
                        "snippet": clean_snip[:500],
                        "published_date": None,
                        "source_provider": "duckduckgo",
                        "verified": True,
                    }
                )
            return results, None
        except Exception as exc:
            return [], f"DuckDuckGo unreachable ({redact_sensitive(str(exc))[:120]})"

    async def _search_wikipedia(self, query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            async with httpx.AsyncClient(timeout=7.0) as client:
                resp = await client.get(
                    f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}&utf8=&format=json&srlimit={max_results}",
                    headers={"User-Agent": "XyvoraAI/2.0"},
                )
            if resp.status_code != 200:
                return [], f"Wikipedia API HTTP {resp.status_code}"
            items = ((resp.json().get("query") or {}).get("search")) or []
            results = []
            for it in items:
                title = it.get("title", "")
                snip = html.unescape(re.sub(r"<[^>]+>", "", it.get("snippet", ""))).strip()
                ts = (it.get("timestamp") or "")[:10] or None
                results.append(
                    {
                        "title": f"{title} — Wikipedia",
                        "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                        "snippet": snip[:500],
                        "published_date": ts,
                        "source_provider": "wikipedia",
                        "verified": True,
                    }
                )
            return results, None
        except Exception as exc:
            return [], f"Wikipedia API unreachable ({redact_sensitive(str(exc))[:120]})"

    async def _search_github_public(self, query: str, max_results: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        clean_q = re.sub(r"[^\w\s\-.+#]", " ", query).strip()
        if not clean_q:
            return [], None
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"https://api.github.com/search/repositories?q={quote_plus(clean_q)}&sort=stars&order=desc&per_page={max_results}",
                    headers={
                        "Accept": "application/vnd.github+json",
                        "User-Agent": "XyvoraAI-Platform/2.0",
                    },
                )
            if resp.status_code != 200:
                return [], f"GitHub API HTTP {resp.status_code}"
            items = resp.json().get("items") or []
            results = []
            for repo in items[:max_results]:
                desc = repo.get("description") or "Public open-source repository."
                stars = repo.get("stargazers_count", 0)
                lang = repo.get("language") or "Multi-language"
                pushed = (repo.get("pushed_at") or repo.get("updated_at") or "")[:10] or None
                results.append(
                    {
                        "title": f"{repo.get('full_name')} ({lang}, ★ {stars:,})",
                        "url": repo.get("html_url", ""),
                        "snippet": f"{desc} | Topics: {', '.join((repo.get('topics') or [])[:6])}".strip(" |"),
                        "published_date": pushed,
                        "source_provider": "github_api",
                        "verified": True,
                    }
                )
            return results, None
        except Exception as exc:
            return [], f"GitHub Search API error ({redact_sensitive(str(exc))[:120]})"

    async def research(
        self,
        query: str,
        mode: str = "quick",
        max_results: int = 5,
        follow_links: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes Quick Search or Deep Multi-Source Research.
        If user supplies explicit http(s) URLs in query, fetches and analyzes those pages directly too!
        """
        t0 = time.perf_counter()
        q = query.strip()
        if not q:
            return {"ok": False, "error": "Research query cannot be empty.", "sources": [], "diagnostics": []}

        is_deep = mode == "deep" or follow_links
        diagnostics: List[str] = []
        sources: List[Dict[str, Any]] = []

        # 0. Direct URL extraction if URL(s) included in query
        explicit_urls = re.findall(r"https?://[^\s<>\"')]+", q)
        for u in explicit_urls[:3]:
            page = await self.fetch_webpage(u)
            if page.get("ok"):
                sources.append(
                    {
                        "title": page["title"],
                        "url": page["url"],
                        "snippet": page["content"][:550],
                        "full_content": page["content"][:5000],
                        "published_date": page.get("published_date"),
                        "source_provider": "direct_fetch",
                        "verified": True,
                        "links": page.get("links", [])[:5],
                    }
                )
            else:
                diagnostics.append(page.get("error", f"Failed fetching {u}"))

        # 1. Try configured commercial search APIs first (Tavily -> Brave)
        search_q = re.sub(r"https?://[^\s<>\"')]+", "", q).strip() or q
        if len(sources) < max_results:
            for fn in (self._search_tavily, self._search_brave):
                hits, diag = await fn(search_q, max_results)
                if diag:
                    diagnostics.append(diag)
                if hits:
                    sources.extend(hits)
                    break

        # 2. Try public search endpoints (DuckDuckGo + Wikipedia + GitHub Public API)
        if len(sources) < max_results:
            ddg_hits, ddg_diag = await self._search_duckduckgo(search_q, max_results)
            if ddg_diag:
                diagnostics.append(ddg_diag)
            sources.extend(ddg_hits)

        if len(sources) < max_results:
            wiki_hits, wiki_diag = await self._search_wikipedia(search_q, max(2, max_results // 2))
            if wiki_diag:
                diagnostics.append(wiki_diag)
            sources.extend(wiki_hits)

        if len(sources) < max_results:
            gh_hits, gh_diag = await self._search_github_public(search_q, max_results)
            if gh_diag:
                diagnostics.append(gh_diag)
            sources.extend(gh_hits)

        # Deduplicate by URL
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for s in sources:
            u = s.get("url") or ""
            if u and u not in seen:
                seen.add(u)
                deduped.append(s)
        sources = deduped[:max_results]

        # 3. Deep Research: fetch readable page content + follow 1 hop of relevant links
        if is_deep and sources:
            for src in sources[:3]:
                if not src.get("full_content"):
                    fetched = await self.fetch_webpage(src["url"], timeout_sec=7.0)
                    if fetched.get("ok") and fetched.get("content"):
                        src["full_content"] = fetched["content"][:4000]
                        if fetched.get("published_date") and not src.get("published_date"):
                            src["published_date"] = fetched["published_date"]
                        # Follow 1 relevant link if available
                        sub_links = fetched.get("links") or []
                        if sub_links and len(sources) < max_results + 2:
                            link_labels = [f"{lk['label']} {lk['url']}" for lk in sub_links]
                            try:
                                lsims = tfidf_cosine_scores(link_labels, search_q, mode="word")
                                best_idx = int(lsims.argmax())
                                if float(lsims[best_idx]) >= 0.18:
                                    sub_url = sub_links[best_idx]["url"]
                                    if sub_url not in seen:
                                        sub_page = await self.fetch_webpage(sub_url, timeout_sec=6.0)
                                        if sub_page.get("ok") and sub_page.get("content"):
                                            seen.add(sub_url)
                                            sources.append(
                                                {
                                                    "title": sub_page["title"],
                                                    "url": sub_page["url"],
                                                    "snippet": sub_page["content"][:450],
                                                    "full_content": sub_page["content"][:3000],
                                                    "published_date": sub_page.get("published_date"),
                                                    "source_provider": "followed_link",
                                                    "verified": True,
                                                }
                                            )
                            except Exception:
                                pass

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        if not sources:
            diag_note = "; ".join(diagnostics[:3]) if diagnostics else "No external search providers responded."
            return {
                "ok": False,
                "mode": mode,
                "query": q,
                "sources": [],
                "citations": [],
                "summary": (
                    f"Web research could not retrieve live results for **\"{q}\"**.\n\n"
                    f"**Provider / Network Diagnostics:** {diag_note}\n\n"
                    f"*Transparency Note:* Xyvora never fabricates web search results or URLs. "
                    f"Ensure outbound internet access is permitted or configure `TAVILY_API_KEY` / `BRAVE_API_KEY` in environment variables."
                ),
                "diagnostics": diagnostics,
                "duration_ms": elapsed_ms,
            }

        # Build structured multi-source synthesis & comparison with numbered citations [1](url)
        citations: List[Dict[str, Any]] = []
        summary_lines = [
            f"### Grounded Web Research ({'Deep Multi-Source' if is_deep else 'Quick Search'} — {len(sources)} Verified Source{'s' if len(sources) != 1 else ''})\n"
        ]

        combined_corpus = []
        dated_count = 0
        providers_used = sorted({s.get("source_provider", "web") for s in sources})

        for idx, s in enumerate(sources, start=1):
            pub = s.get("published_date")
            if pub:
                dated_count += 1
            date_tag = f" *(Updated/Published: `{pub}`)*" if pub else " *(Publication date not stated)*"
            excerpt = (s.get("full_content") or s.get("snippet") or "").strip()
            short_excerpt = re.sub(r"\s+", " ", excerpt)[:340]
            combined_corpus.append(excerpt)

            citations.append(
                {
                    "index": idx,
                    "title": s["title"],
                    "url": s["url"],
                    "published_date": pub,
                    "provider": s.get("source_provider", "web"),
                    "snippet": short_excerpt,
                    "verified": True,
                }
            )
            summary_lines.append(
                f"{idx}. **[{s['title']}]({s['url']})**{date_tag}\n   {short_excerpt}"
            )

        shared_themes = extract_top_keyphrases(" ".join(combined_corpus), top_k=6)
        if shared_themes:
            summary_lines.append(
                "\n**Cross-Source Comparison & Themes:** Common concepts across retrieved sources include "
                + ", ".join(f"`{t}`" for t in shared_themes)
                + "."
            )

        uncertainty_note = (
            f"**Verification Status:** All {len(sources)} citation URL(s) above were retrieved live via `{', '.join(providers_used)}` "
            f"({dated_count}/{len(sources)} include explicit timestamps)."
        )
        if diagnostics:
            uncertainty_note += f" *(Note: Some secondary search endpoints were unreachable in this network: {'; '.join(diagnostics[:2])})*"
        summary_lines.append("\n" + uncertainty_note)

        return {
            "ok": True,
            "mode": mode,
            "query": q,
            "sources": sources,
            "citations": citations,
            "themes": shared_themes,
            "summary": "\n\n".join(summary_lines),
            "diagnostics": diagnostics,
            "duration_ms": elapsed_ms,
        }
