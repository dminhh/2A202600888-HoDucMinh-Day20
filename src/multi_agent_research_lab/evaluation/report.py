"""Benchmark report rendering."""

from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics], query: str = "") -> str:
    """Render benchmark metrics to a markdown report."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        f"**Date:** {now}  ",
        f"**Query:** {query}  " if query else "",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for m in metrics:
        cost = f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd is not None else "—"
        quality = f"{m.quality_score:.1f}" if m.quality_score is not None else "—"
        lines.append(f"| {m.run_name} | {m.latency_seconds:.2f} | {cost} | {quality} | {m.notes} |")

    if len(metrics) == 2:  # noqa: PLR2004
        a, b = metrics[0], metrics[1]
        lines += [
            "",
            "## Comparison",
            "",
        ]
        if a.latency_seconds and b.latency_seconds:
            slower = b.run_name if b.latency_seconds > a.latency_seconds else a.run_name
            ratio = max(a.latency_seconds, b.latency_seconds) / min(a.latency_seconds, b.latency_seconds)
            lines.append(f"- **Latency:** `{slower}` is {ratio:.1f}× slower")

        if a.estimated_cost_usd and b.estimated_cost_usd:
            pricier = b.run_name if b.estimated_cost_usd > a.estimated_cost_usd else a.run_name
            ratio = max(a.estimated_cost_usd, b.estimated_cost_usd) / min(a.estimated_cost_usd, b.estimated_cost_usd)
            lines.append(f"- **Cost:** `{pricier}` costs {ratio:.1f}× more")

        if a.quality_score is not None and b.quality_score is not None:
            better = b.run_name if b.quality_score > a.quality_score else a.run_name
            diff = abs(b.quality_score - a.quality_score)
            lines.append(f"- **Quality:** `{better}` scores {diff:.1f} pts higher")

    lines += [
        "",
        "## Failure Modes & Fixes",
        "",
        "| Observed | Root Cause | Fix Applied |",
        "|---|---|---|",
        "| Mock sources (no TAVILY_API_KEY) | Search client falls back to static snippets | Set `TAVILY_API_KEY` in `.env` |",
        "| LLM routing occasionally redundant | Supervisor calls LLM even for deterministic cases | Deterministic fallback already in place |",
        "",
        "## Notes",
        "",
        "- Quality score is a heuristic (word count + headings + citations + sources). "
        "Replace with an LLM-as-judge scorer for production.",
        "- Cost estimate uses gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output).",
        "- Single-agent baseline does not record token usage in `agent_results`; cost shown as `—`.",
        "",
    ]
    return "\n".join(line for line in lines if line is not None)
