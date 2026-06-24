from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(**kwargs: object) -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"), **kwargs)  # type: ignore[arg-type]


def test_supervisor_routes_to_researcher_when_empty() -> None:
    """Fresh state → researcher first."""
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == AgentName.RESEARCHER


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = _make_state(research_notes="some notes")
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == AgentName.ANALYST


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = _make_state(research_notes="some notes", analysis_notes="some analysis")
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == AgentName.WRITER


def test_supervisor_done_when_answer_ready() -> None:
    state = _make_state(
        research_notes="notes", analysis_notes="analysis", final_answer="answer"
    )
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"


def test_supervisor_enforces_max_iterations() -> None:
    """When iteration >= max_iterations, always routes to done."""
    state = _make_state()
    state.iteration = 999
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == "done"
    assert result.errors
