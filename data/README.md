# MemoryStress Dataset

The full benchmark dataset (2.6 MB, 583 facts, 1000 sessions, 300 questions) is hosted on HuggingFace:

**https://huggingface.co/datasets/singularityjason/memorystress**

## Quick Download

```bash
# Via huggingface_hub
pip install huggingface_hub
huggingface-cli download singularityjason/memorystress --local-dir data/

# Or direct download
wget https://huggingface.co/datasets/singularityjason/memorystress/resolve/main/memorystress_v1.json -O data/memorystress_v1.json
```

## Generate Your Own

You can also generate a fresh dataset (requires an OpenAI API key, ~$5):

```bash
python scripts/generate.py --model gpt-4o --seed 42 --output data/memorystress_v1.json --validate
```

Or a small test dataset (free with `--no-llm`):

```bash
python scripts/generate.py --small --no-llm --output data/test.json
```

## Dataset Format

```json
{
  "version": "1.0",
  "phases": { "1": {...}, "2": {...}, "3": {...} },
  "stats": { "total_facts": 583, "total_sessions": 1000, ... },
  "facts": [ { "fact_id": "F001", "content": "...", ... } ],
  "contradiction_chains": [ { "chain_id": "C001", ... } ],
  "sessions": [ { "session_id": "S0001", "turns": [...], ... } ],
  "questions": [ { "question_id": "Q001", "question": "...", ... } ]
}
```
