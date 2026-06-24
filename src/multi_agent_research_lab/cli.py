"""Command-line entrypoint for the lab starter."""

import pathlib
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    try:
        llm = LLMClient()
        response = llm.complete(
            system_prompt="You are a research assistant. Write a thorough, well-structured answer.",
            user_prompt=request.query,
        )
        state.final_answer = response.content
        token_info = f"[dim]tokens: {response.input_tokens} in / {response.output_tokens} out[/dim]"
        console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
        console.print(token_info)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Config Error", style="red"))
        raise typer.Exit(code=1) from exc


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art and write a 500-word summary"
    ),
    output: Annotated[pathlib.Path, typer.Option("--output", "-o", help="Save report to file")] = pathlib.Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Run single-agent and multi-agent, then compare side-by-side."""

    _init()
    llm = LLMClient()

    # ── single-agent runner ────────────────────────────────────────────────
    def baseline_runner(q: str) -> ResearchState:
        req = ResearchQuery(query=q)
        state = ResearchState(request=req)
        response = llm.complete(
            system_prompt="You are a research assistant. Write a thorough, well-structured answer.",
            user_prompt=q,
        )
        state.final_answer = response.content
        return state

    # ── multi-agent runner ─────────────────────────────────────────────────
    def multi_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(state)

    console.rule("[bold]Running single-agent baseline[/bold]")
    _, m_baseline = run_benchmark("single-agent", query, baseline_runner)

    console.rule("[bold]Running multi-agent workflow[/bold]")
    _, m_multi = run_benchmark("multi-agent", query, multi_runner)

    report_md = render_markdown_report([m_baseline, m_multi], query=query)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_md)

    console.print()
    console.print(Markdown(report_md))
    console.print(f"\n[green]Report saved → {output}[/green]")


if __name__ == "__main__":
    app()
