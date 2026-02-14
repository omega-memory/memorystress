# MemoryStress Benchmark Report

- **Adapter**: omega
- **Model**: gpt-4o
- **Date**: 2026-02-14 18:20
- **Overall**: 98/300 (32.7%)

## Degradation Curve

| Phase | Accuracy |
|-------|----------|
| 1 | 31.4% ###### |
| 2 | 42.4% ######## |
| 3 | 27.3% ##### |
| 4 | 25.5% ##### |

**Degradation slope**: -0.0328 per phase

## Recall by Age

| Age Bucket | Accuracy |
|------------|----------|
| 0-100 sessions | 33.8% |
| 100-500 sessions | 39.7% |
| 500-1000 sessions | 21.3% |

## Scoring Dimensions

| Dimension | Score |
|-----------|-------|
| Contradiction Resolution | 21.4% (56 Qs) |
| Cross-Agent Recall | 31.2% (32 Qs) |
| Cold Start Recovery | 0.0% (0 Qs) |

## Cost Efficiency

| Phase | $/Correct Answer |
|-------|-----------------|
| 1 | $0.0311 |
| 2 | $0.0337 |
| 3 | $0.0655 |
| 4 | $0.0292 |

## Per-Type Breakdown

| Question Type | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| cold_start_recall | 6 | 16 | 37.5% |
| contradiction_resolution | 12 | 56 | 21.4% |
| cross_agent_recall | 10 | 32 | 31.2% |
| fact_recall | 30 | 80 | 37.5% |
| preference_recall | 13 | 35 | 37.1% |
| single_mention_recall | 13 | 47 | 27.7% |
| temporal_ordering | 14 | 34 | 41.2% |
