# Contributing to MemoryStress

We welcome contributions! Here's how you can help.

## Add Your Memory System to the Leaderboard

1. Implement the `MemorySystemAdapter` interface (see [adapters/](memorystress/adapters/) for examples)
2. Run the benchmark:
   ```bash
   python scripts/run.py --dataset data/memorystress_v1.json --adapter your_system --grade --output-dir results/your_system
   ```
3. Open a PR with your `results/your_system/` directory

Works with any memory backend: OMEGA, Mem0, Zep, LangMem, OpenAI Assistants, MCP servers, custom RAG pipelines, vector databases, knowledge graphs.

## Improve the Benchmark

- Add new question types or contradiction patterns
- Enhance grading prompts for better accuracy
- Add support for new LLM providers
- Improve dataset generation

## Code Style

- Python 3.11+
- Type hints on all public functions
- Format with `black`

## Questions?

Open an [issue](https://github.com/omega-memory/memorystress/issues) or [discussion](https://github.com/omega-memory/memorystress/discussions).
