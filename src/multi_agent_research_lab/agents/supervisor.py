"""Supervisor / router — decides which worker runs next and when to stop."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research workflow supervisor. Given the current state of a research task,
decide which agent should run next.

Agents available:
- researcher  : gathers raw information and fills research_notes
- analyst     : analyses research_notes and fills analysis_notes
- writer      : synthesises everything into final_answer
- done        : workflow is complete; final_answer is ready

Rules:
1. Always start with researcher if research_notes is empty.
2. Move to analyst once research_notes is present and analysis_notes is empty.
3. Move to writer once analysis_notes is present and final_answer is empty.
4. Return done once final_answer is present.
5. If something looks incomplete, you may repeat a step (researcher or analyst) once.
6. Never exceed the iteration limit.

Reply with ONLY one word — the agent name: researcher, analyst, writer, or done.
"""


def _deterministic_route(state: ResearchState) -> str:
    """Rule-based fallback routing used when LLM is unavailable."""
    if state.research_notes is None:
        return AgentName.RESEARCHER
    if state.analysis_notes is None:
        return AgentName.ANALYST
    if state.final_answer is None:
        return AgentName.WRITER
    return "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self._settings = get_settings()
        try:
            self._llm = LLMClient()
            self._use_llm = True
        except AgentExecutionError:
            logger.warning("LLM unavailable — supervisor will use deterministic routing")
            self._llm = None  # type: ignore[assignment]
            self._use_llm = False

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect state, pick next route, enforce max_iterations."""

        if state.iteration >= self._settings.max_iterations:
            logger.warning("Max iterations (%d) reached — forcing done", self._settings.max_iterations)
            state.errors.append(f"Stopped after {state.iteration} iterations (max_iterations limit).")
            state.record_route("done")
            state.add_trace_event("supervisor", {"decision": "done", "reason": "max_iterations"})
            return state

        route = self._decide(state)

        logger.info("Supervisor → %s (iteration %d)", route, state.iteration)
        state.record_route(route)
        state.add_trace_event(
            "supervisor",
            {
                "decision": route,
                "iteration": state.iteration,
                "has_research": state.research_notes is not None,
                "has_analysis": state.analysis_notes is not None,
                "has_answer": state.final_answer is not None,
            },
        )
        return state

    def _decide(self, state: ResearchState) -> str:
        """Return next agent name, falling back to deterministic routing on error."""
        if not self._use_llm:
            return _deterministic_route(state)

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Iteration: {state.iteration}\n"
            f"research_notes present: {state.research_notes is not None}\n"
            f"analysis_notes present: {state.analysis_notes is not None}\n"
            f"final_answer present: {state.final_answer is not None}\n"
            f"recent errors: {state.errors[-3:] if state.errors else 'none'}\n"
            "Which agent should run next?"
        )
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            route = response.content.strip().lower().split()[0]
            valid = {AgentName.RESEARCHER, AgentName.ANALYST, AgentName.WRITER, "done"}
            if route not in valid:
                logger.warning("LLM returned invalid route %r — using deterministic fallback", route)
                route = _deterministic_route(state)
        except Exception as exc:
            logger.warning("LLM routing failed (%s) — using deterministic fallback", exc)
            route = _deterministic_route(state)

        return route
