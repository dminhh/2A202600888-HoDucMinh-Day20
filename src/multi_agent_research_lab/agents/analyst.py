"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a critical analyst. Given raw research notes, produce a structured analysis in markdown:

## Key Claims
List the most important claims or findings, each with a confidence label (High / Medium / Low).

## Strengths of the Evidence
What is well-supported across multiple sources?

## Gaps & Weak Evidence
What is asserted but poorly evidenced, or where do sources conflict?

## Implications
What does this mean for the stated audience?

Be precise and critical. Do not repeat the research notes verbatim.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Analyse research_notes and populate analysis_notes."""

        if not state.research_notes:
            raise AgentExecutionError("AnalystAgent requires research_notes — run ResearcherAgent first")

        logger.info("AnalystAgent: analysing research notes")
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}"
        )

        response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event("analyst", {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        })
        logger.info("AnalystAgent: done (%s out tokens)", response.output_tokens)
        return state
