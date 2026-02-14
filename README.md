# MemoryStress

**The first longitudinal benchmark for AI memory systems.**

Every existing memory benchmark tests recall from a handful of sessions. MemoryStress tests what happens at session 1,000 — when facts contradict, memories fade, and noise accumulates over 10 simulated months.

```
583 facts  ×  1,000 sessions  ×  300 questions  ×  6 scoring dimensions
```

## Why MemoryStress?

Current benchmarks don't test what breaks in production:

| Benchmark | Sessions | Contradictions | Degradation Curve | Multi-Agent |
|-----------|:--------:|:--------------:|:-----------------:|:-----------:|
| LongMemEval | ~40 | No | No | No |
| MemoryAgentBench | Short | No | No | No |
| BEAM | Synthetic | No | No | No |
| **MemoryStress** | **1,000** | **Yes (40 chains)** | **Yes (4 checkpoints)** | **Yes** |

Real agents run for months. Their memory systems face:
- **Accumulation pressure**: 1,000 sessions of growing noise
- **Contradiction resolution**: Facts that change, revert, and partially update
- **Degradation over time**: Can you still find a fact from session 12 after ingesting session 997?
- **Cross-agent recall**: Information shared across agent identities
- **Cold start recovery**: Fresh agent instance accessing an existing memory store

MemoryStress measures all of these.

## Leaderboard

| System | Model | Score | Contradiction | Degradation Slope | Cost/Correct | Details |
|--------|-------|:-----:|:-------------:|:-----------------:|:------------:|---------|
| **OMEGA v1.0** | GPT-4o | **32.7%** (98/300) | 21.4% | -0.033 | $0.041 | [results](results/omega_v5/) |
| Raw Context | GPT-4o | TBD | TBD | TBD | TBD | — |
| Mem0 | GPT-4o | TBD | TBD | TBD | TBD | — |
| Null (baseline) | — | 0% | 0% | 0.000 | — | [results](results/null/) |

> **32.7% is a strong result on an intentionally brutal benchmark.** The null baseline scores 0%. A context-window approach hits its token ceiling around session 200. This benchmark asks about single-mention facts buried in noisy conversations from hundreds of sessions ago.

**Submit your results**: Run the benchmark, open a PR with your `results/` directory.

## Quick Start

```bash
# Install
pip install memorystress

# Download dataset
# Option A: From HuggingFace
pip install huggingface_hub
huggingface-cli download singularityjason/memorystress --local-dir data/

# Option B: Generate your own (~$5 in API costs)
python scripts/generate.py --model gpt-4o --seed 42 --output data/memorystress_v1.json

# Run the null baseline (free, instant)
python scripts/run.py --dataset data/memorystress_v1.json --adapter null --grade

# Run with OMEGA (requires: pip install omega-memory)
python scripts/run.py --dataset data/memorystress_v1.json --adapter omega --grade \
  --extract-facts --query-augment --output-dir results/omega_v5

# Run with Mem0 (requires: pip install mem0ai)
python scripts/run.py --dataset data/memorystress_v1.json --adapter mem0 --grade \
  --output-dir results/mem0
```

## How It Works

### 1. Dataset Generation

A GPT-4o pipeline generates:
- **583 facts** across 6 categories (preferences, decisions, technical facts, personal info, events, relationships)
- **1,000 conversation sessions** across 3 phases of increasing noise
- **40 contradiction chains** where facts update, revert, accumulate, or partially change
- **300 questions** at 4 phase checkpoints, spanning 7 question types

### 2. Three Phases of Pressure

```
Phase 1 (Sessions 1-100)     Low noise, establishing baseline
Phase 2 (Sessions 101-500)   Growing noise, contradictions emerge
Phase 3 (Sessions 501-1000)  Dense, high-entropy, multi-topic stress
Phase 4 (Recovery)           No new sessions — can you still recall?
```

### 3. Six Scoring Dimensions

| Dimension | What It Measures |
|-----------|-----------------|
| **Recall@Age** | Accuracy by fact age (0-100, 100-500, 500-1000 sessions) |
| **Degradation Curve** | Accuracy at each phase checkpoint (slope = degradation rate) |
| **Contradiction Resolution** | Can the system identify the most recent value? |
| **Cost Efficiency** | Dollars per correct answer at each phase |
| **Cross-Agent Recall** | Information recall across agent identities |
| **Cold Start Recovery** | Fresh agent accessing an existing memory store |

### 4. Seven Question Types

- Fact recall, Preference recall, Contradiction resolution, Temporal ordering, Cross-agent recall, Cold start recall, Single-mention recall

## Add Your Memory System

Implement 4 methods:

```python
from memorystress.adapters.base import MemorySystemAdapter, IngestResult, QueryResult, CostSnapshot

class MyAdapter(MemorySystemAdapter):
    def ingest(self, session: dict) -> IngestResult:
        """Store a conversation session."""
        # session has: session_id, turns, simulated_date, agent_id, phase
        ...

    def query(self, question: dict) -> QueryResult:
        """Answer a question from memory."""
        # question has: question, agent_scope, question_type
        ...

    def reset(self) -> None:
        """Clear all stored data."""
        ...

    def get_cost(self) -> CostSnapshot:
        """Return cumulative token/cost accounting."""
        ...
```

Then add it to `scripts/run.py`'s `create_adapter()` factory and run the benchmark.

## OMEGA's Results: Deep Dive

### Degradation Curve

```
Phase 1 (100 sessions):   31.4%  ██████
Phase 2 (500 sessions):   42.4%  ████████    ← peak: more data helps retrieval
Phase 3 (1000 sessions):  27.3%  █████       ← noise dilution, not data loss
Phase 4 (recovery):       25.5%  █████
```

### Per-Type Breakdown

| Question Type | Score | N |
|---------------|:-----:|:-:|
| Temporal ordering | 41.2% | 34 |
| Fact recall | 37.5% | 80 |
| Cold start recall | 37.5% | 16 |
| Preference recall | 37.1% | 35 |
| Cross-agent recall | 31.2% | 32 |
| Single-mention recall | 27.7% | 47 |
| Contradiction resolution | 21.4% | 56 |

### Recall by Fact Age

| Age | Score |
|-----|:-----:|
| 0-100 sessions | 33.8% |
| 100-500 sessions | 39.7% |
| 500-1000 sessions | 21.3% |

Mid-range facts (100-500 sessions old) are easiest — enough time to be reinforced, not so old that noise drowns them.

## Comparison with Other Benchmarks

| Feature | MemoryStress | LongMemEval | MemoryAgentBench | BEAM |
|---------|:------------:|:-----------:|:----------------:|:----:|
| Session count | 1,000 | ~40 | Short | Synthetic |
| Degradation measurement | Yes | No | No | No |
| Contradiction chains | 40 | No | No | No |
| Multi-agent | Yes | No | No | No |
| Cold start testing | Yes | No | No | No |
| Cost tracking | Yes | No | No | No |
| Temporal questions | Yes | Yes | No | No |
| Question types | 7 | 4 | Task-based | 3 |

## Dataset

The full dataset is available on HuggingFace: [singularityjason/memorystress](https://huggingface.co/datasets/singularityjason/memorystress)

See [data/README.md](data/README.md) for format details and generation instructions.

## Citation

```bibtex
@software{memorystress2026,
  title={MemoryStress: The First Longitudinal Benchmark for AI Memory Systems},
  author={OMEGA Memory},
  url={https://github.com/omega-memory/memorystress},
  year={2026}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
