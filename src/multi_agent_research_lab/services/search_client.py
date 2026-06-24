"""Search client — Tavily when key is set, mock otherwise."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily if TAVILY_API_KEY is configured; falls back to a keyword mock
    so the workflow runs end-to-end without a search key.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._use_tavily = bool(self._settings.tavily_api_key)
        if self._use_tavily:
            from tavily import TavilyClient  # type: ignore[import-untyped]
            self._tavily = TavilyClient(api_key=self._settings.tavily_api_key)
        logger.debug("SearchClient mode: %s", "tavily" if self._use_tavily else "mock")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if self._use_tavily:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        response = self._tavily.search(query=query, max_results=max_results)
        return [
            SourceDocument(
                title=r.get("title", "Untitled"),
                url=r.get("url"),
                snippet=r.get("content", ""),
            )
            for r in response.get("results", [])
        ]

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        logger.info("SearchClient: no TAVILY_API_KEY — returning mock sources")
        mock_snippets = [
            (
                "Overview of the topic",
                f"Mock source 1: {query} is an active research area combining "
                "multiple disciplines. Recent surveys highlight key challenges.",
            ),
            (
                "Key methods and approaches",
                f"Mock source 2: Leading approaches to {query} include neural, "
                "symbolic, and hybrid methods. Benchmarks show steady improvement.",
            ),
            (
                "State-of-the-art results",
                f"Mock source 3: SOTA results on {query} as of 2024 demonstrate "
                "significant gains over baselines on standard evaluation sets.",
            ),
        ]
        return [
            SourceDocument(title=title, snippet=snippet)
            for title, snippet in mock_snippets[:max_results]
        ]
