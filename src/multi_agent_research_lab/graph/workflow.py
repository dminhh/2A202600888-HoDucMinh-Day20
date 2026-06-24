"""LangGraph multi-agent workflow."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# ── node names ────────────────────────────────────────────────────────────────
_SUPERVISOR = "supervisor"
_RESEARCHER = "researcher"
_ANALYST = "analyst"
_WRITER = "writer"


def _wrap(agent_cls: type, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Instantiate agent, run it, return updated state dict."""
    state = ResearchState(**state_dict)
    result = agent_cls().run(state)
    return result.model_dump()


class MultiAgentWorkflow:
    """Builds and runs the Supervisor → Researcher → Analyst → Writer graph."""

    def build(self) -> Any:
        """Compile LangGraph StateGraph with conditional routing."""

        graph = StateGraph(dict)

        # ── nodes ──────────────────────────────────────────────────────────
        graph.add_node(_SUPERVISOR, lambda s: _wrap(SupervisorAgent, s))
        graph.add_node(_RESEARCHER, lambda s: _wrap(ResearcherAgent, s))
        graph.add_node(_ANALYST,    lambda s: _wrap(AnalystAgent,    s))
        graph.add_node(_WRITER,     lambda s: _wrap(WriterAgent,     s))

        # ── entry ──────────────────────────────────────────────────────────
        graph.add_edge(START, _SUPERVISOR)

        # ── conditional routing from supervisor ────────────────────────────
        def route(state: dict[str, Any]) -> str:
            history: list[str] = state.get("route_history", [])
            last = history[-1] if history else "done"
            if last == _RESEARCHER:
                return _RESEARCHER
            if last == _ANALYST:
                return _ANALYST
            if last == _WRITER:
                return _WRITER
            return END  # "done" or unknown

        graph.add_conditional_edges(
            _SUPERVISOR,
            route,
            {
                _RESEARCHER: _RESEARCHER,
                _ANALYST:    _ANALYST,
                _WRITER:     _WRITER,
                END:         END,
            },
        )

        # ── each worker loops back to supervisor ───────────────────────────
        graph.add_edge(_RESEARCHER, _SUPERVISOR)
        graph.add_edge(_ANALYST,    _SUPERVISOR)
        graph.add_edge(_WRITER,     _SUPERVISOR)

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile graph, invoke it, return final ResearchState."""

        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            compiled = self.build()
            logger.info("Workflow starting | query=%r", state.request.query)

            try:
                result_dict = compiled.invoke(state.model_dump())
            except Exception as exc:
                raise AgentExecutionError(f"Workflow failed: {exc}") from exc

            final = ResearchState(**result_dict)
            span["attributes"]["iterations"] = final.iteration
            span["attributes"]["route_history"] = final.route_history

            logger.info(
                "Workflow done | iterations=%d route=%s",
                final.iteration,
                " → ".join(final.route_history),
            )

        return final
