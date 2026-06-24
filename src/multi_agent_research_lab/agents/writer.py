"""Writer agent — synthesises research + analysis into final answer."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical writer producing a polished, audience-appropriate response.

Guidelines:
- Open with a clear, direct answer to the query.
- Use the research and analysis notes as your source of truth.
- Structure the response with markdown headings.
- Cite key sources inline using [Source Title] notation where relevant.
- Close with a brief "Further Reading" section listing the source titles.
- Match the depth and vocabulary to the stated audience.
- Do NOT fabricate facts not present in the notes.
"""


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesise all notes into state.final_answer."""

        if not state.research_notes:
            raise AgentExecutionError("WriterAgent requires research_notes")
        if not state.analysis_notes:
            raise AgentExecutionError("WriterAgent requires analysis_notes")

        logger.info("WriterAgent: synthesising final answer")

        source_titles = "\n".join(
            f"- {s.title}" + (f" ({s.url})" if s.url else "")
            for s in state.sources
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"## Research Notes\n{state.research_notes}\n\n"
            f"## Analysis Notes\n{state.analysis_notes}\n\n"
            f"## Available Sources\n{source_titles or 'None'}"
        )

        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.final_answer = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event("writer", {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })
        logger.info("WriterAgent: done (%s out tokens)", response.output_tokens)
        return state
