"""MemoryStress — the first longitudinal benchmark for AI memory systems.

Evaluate AI agent memory, LLM memory retention, and RAG pipeline degradation
across 1,000 sessions. Works with OMEGA, Mem0, Zep, LangMem, OpenAI Assistants,
MCP servers, and custom memory backends.

Supports OpenAI GPT-4o, Anthropic Claude, Google Gemini, Groq, and Grok.

    >>> from memorystress.dataset import DatasetLoader
    >>> loader = DatasetLoader("data/memorystress_v1.json")
    >>> data = loader.load()

https://github.com/omega-memory/memorystress
"""

__version__ = "0.1.0"
