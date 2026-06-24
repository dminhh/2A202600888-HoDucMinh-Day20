"""Researcher agent — gathers sources and writes research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research specialist. Given a query and a list of source snippets,
write concise, factual research notes in markdown. Cover:
- Key definitions and background
- Main findings from the sources
- Important authors, methods, or benchmarks mentioned
- Open questions or gaps

Be objective. Cite sources by their title in brackets, e.g. [Overview of the topic].
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Search for sources, then ask LLM to synthesise research notes."""

        logger.info("ResearcherAgent: searching for '%s'", state.request.query)
        try:
            sources = self._search.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
        except Exception as exc:
            raise AgentExecutionError(f"Search failed: {exc}") from exc

        state.sources.extend(sources)
        state.add_trace_event("researcher", {"sources_found": len(sources)})

        snippets = "\n\n".join(
            f"[{s.title}]\n{s.snippet}" for s in sources
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"Sources:\n{snippets}"
        )

        logger.info("ResearcherAgent: generating research notes")
        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        logger.info("ResearcherAgent: done (%s sources, %s out tokens)",
                    len(sources), response.output_tokens)
        return state
