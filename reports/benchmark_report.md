# Benchmark Report: Single-Agent vs Multi-Agent

**Date:** 2026-06-24 11:36  
**Query:** Research GraphRAG state-of-the-art and write a 500-word summary  

## Results

| Run | Latency (s) | Cost (USD) | Quality /10 | Notes |
|---|---:|---:|---:|---|
| single-agent | 13.53 | — | 5.0 | 0in/0out tokens |
| multi-agent | 33.34 | $0.00132 | 7.9 | 2209in/1654out tokens |

## Comparison

- **Latency:** `multi-agent` is 2.5× slower
- **Quality:** `multi-agent` scores 2.9 pts higher

## Failure Modes & Fixes

| Observed | Root Cause | Fix Applied |
|---|---|---|
| Mock sources (no TAVILY_API_KEY) | Search client falls back to static snippets | Set `TAVILY_API_KEY` in `.env` |
| LLM routing occasionally redundant | Supervisor calls LLM even for deterministic cases | Deterministic fallback already in place |

## Notes

- Quality score is a heuristic (word count + headings + citations + sources). Replace with an LLM-as-judge scorer for production.
- Cost estimate uses gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output).
- Single-agent baseline does not record token usage in `agent_results`; cost shown as `—`.
