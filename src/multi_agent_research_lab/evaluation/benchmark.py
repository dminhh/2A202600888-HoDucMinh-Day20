"""Benchmark: single-agent vs multi-agent comparison."""

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

# Cost per 1 000 tokens (gpt-4o-mini, as of 2024)
_COST_PER_1K_INPUT_USD = 0.000150
_COST_PER_1K_OUTPUT_USD = 0.000600


def _total_tokens(state: ResearchState) -> tuple[int, int]:
    """Sum input/output tokens across all agent_results."""
    inp = sum(r.metadata.get("input_tokens") or 0 for r in state.agent_results)
    out = sum(r.metadata.get("output_tokens") or 0 for r in state.agent_results)
    return inp, out


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1000 * _COST_PER_1K_INPUT_USD
        + output_tokens / 1000 * _COST_PER_1K_OUTPUT_USD
    )


def _quality_score(state: ResearchState) -> float:
    """Heuristic quality score 0-10 based on structural completeness."""
    if not state.final_answer:
        return 0.0
    answer = state.final_answer

    score = 0.0
    # length — penalise very short answers
    word_count = len(answer.split())
    score += min(word_count / 100, 3.0)          # up to 3 pts for 300+ words

    # structure — markdown headings
    heading_count = answer.count("\n#")
    score += min(heading_count * 0.5, 2.0)        # up to 2 pts

    # citations — [Source Title] pattern
    import re
    citations = len(re.findall(r"\[[^\]]+\]", answer))
    score += min(citations * 0.5, 2.0)            # up to 2 pts

    # further reading section
    if "further reading" in answer.lower():
        score += 1.0

    # sources available
    score += min(len(state.sources) * 0.3, 1.0)  # up to 1 pt

    return round(min(score, 10.0), 2)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Time a runner, compute cost estimate and heuristic quality score."""

    logger.info("Benchmark starting: %s", run_name)
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    inp, out = _total_tokens(state)
    cost = _estimate_cost(inp, out) if (inp or out) else None
    quality = _quality_score(state)

    notes_parts = [f"{inp}in/{out}out tokens"]
    if state.errors:
        notes_parts.append(f"{len(state.errors)} error(s)")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(cost, 6) if cost else None,
        quality_score=quality,
        notes=", ".join(notes_parts),
    )
    logger.info(
        "Benchmark done: %s | latency=%.2fs quality=%.1f/10 cost=$%.5f",
        run_name, latency, quality, cost or 0,
    )
    return state, metrics
