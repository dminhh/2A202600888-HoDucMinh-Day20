# Benchmark Report: Single-Agent vs Multi-Agent

**Date:** 2026-06-24 11:55  
**Query:** Research GraphRAG state-of-the-art and write a 500-word summary  

## Results

| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |
|---|---:|---:|---:|---|
| single-agent | 12.57 | — | 5.0 | 0in/0out tokens |
| multi-agent | 41.41 | $0.00135 | 8.9 | 2223in/1686out tokens |

## Comparison

- **Latency:** `multi-agent` is 3.3× slower
- **Quality:** `multi-agent` scores 3.9 pts higher

## Failure Modes & Fixes

| Observed | Root Cause | Fix Applied |
|---|---|---|
| Mock sources (no TAVILY_API_KEY) | Search client falls back to static snippets | Set `TAVILY_API_KEY` in `.env` |
| LLM routing occasionally redundant | Supervisor calls LLM even for deterministic cases | Deterministic fallback already in place |

## Notes

- Quality score is a heuristic (word count + headings + citations + sources). Replace with an LLM-as-judge scorer for production.
- Cost estimate uses gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output).
- Single-agent baseline does not record token usage in `agent_results`; cost shown as `—`.
