"""Mem0 adapter — uses mem0ai pip package for memory storage and retrieval.

Mem0 stores memories as extracted facts using its built-in LLM pipeline.
Search returns memory snippets which are then passed to an LLM to generate
an answer (same pattern as the OMEGA adapter's RAG pipeline).

Requires: pip install mem0ai
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

from memorystress.adapters.base import (
    CostSnapshot,
    IngestResult,
    MemorySystemAdapter,
    QueryResult,
)
from memorystress.llm import call_llm

try:
    from mem0 import Memory
except ImportError:
    Memory = None

_RAG_PROMPT = """\
I will give you several memory snippets from past conversations. \
Please answer the question based on the relevant memories. \
If the question cannot be answered based on the provided memories, say so.

CRITICAL RULES:
1. Memories are provided in no particular order. Use all relevant ones.
2. When multiple memories discuss the same topic but give DIFFERENT information, \
prefer the memory that seems most recent or most specific.
3. Give a direct, concise answer.

Memories:

{memories}

Question: {question}
Answer:"""


def _format_turns(turns: list[dict]) -> str:
    """Format session turns as message dicts for Mem0 ingestion."""
    lines = []
    for turn in turns:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines)


class Mem0Adapter(MemorySystemAdapter):
    """Adapter wrapping Mem0 (mem0ai) for benchmarking.

    Uses local Qdrant (embedded, no server needed) + OpenAI for embeddings
    and fact extraction. Each benchmark run gets a fresh data directory.

    Requires: pip install mem0ai
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        retrieval_limit: int = 20,
    ):
        if Memory is None:
            raise ImportError(
                "Mem0 adapter requires mem0ai. "
                "Install it with: pip install mem0ai"
            )
        self.model = model
        self.api_key = api_key
        self.retrieval_limit = retrieval_limit
        self._agent_id: str = "agent_a"
        self._tmpdir: str | None = None
        self._memory = None
        self._cost = CostSnapshot()
        self._init_memory()

    def _init_memory(self) -> None:
        """Create a fresh temp directory and Mem0 Memory instance."""
        self._tmpdir = tempfile.mkdtemp(prefix="memorystress_mem0_")

        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memorystress",
                    "path": self._tmpdir,
                },
            },
        }
        self._memory = Memory.from_config(config)

    def set_agent_id(self, agent_id: str) -> None:
        self._agent_id = agent_id

    def ingest(self, session: dict) -> IngestResult:
        content = _format_turns(session.get("turns", []))
        if not content.strip():
            return IngestResult(success=True)

        session_id = session.get("session_id", "")
        agent_id = session.get("agent_id", self._agent_id)

        try:
            # Mem0's add() with infer=True extracts facts via LLM
            self._memory.add(
                content,
                user_id=agent_id,
                metadata={"session_id": session_id},
            )

            # Estimate tokens (Mem0 uses ~1 LLM call for fact extraction
            # + 1 embedding call per add)
            tokens = len(content) // 4 + 200  # extraction overhead
            self._cost.ingest_tokens += tokens
            self._cost.total_tokens += tokens
            self._cost.total_api_calls += 2  # 1 LLM + 1 embedding

            return IngestResult(success=True, tokens_used=tokens, api_calls=2)
        except Exception as e:
            return IngestResult(success=False, error=str(e))

    def query(self, question: dict) -> QueryResult:
        question_text = question.get("question", "")
        agent_scope = question.get("agent_scope", self._agent_id)

        t0 = time.monotonic()

        try:
            # Search returns memory snippets, not generated answers
            search_results = self._memory.search(
                question_text,
                user_id=agent_scope,
                limit=self.retrieval_limit,
            )
            results = search_results.get("results", []) if isinstance(search_results, dict) else search_results
        except Exception:
            results = []

        retrieval_ms = (time.monotonic() - t0) * 1000

        # Format memory snippets for RAG
        memory_blocks = []
        context_texts = []
        for i, mem in enumerate(results, 1):
            if isinstance(mem, dict):
                text = mem.get("memory", "")
            else:
                text = str(mem)
            if text:
                memory_blocks.append(f"[Memory {i}] {text}")
                context_texts.append(text[:200])

        memories_str = "\n\n".join(memory_blocks) if memory_blocks else "(No relevant memories found)"
        prompt = _RAG_PROMPT.format(memories=memories_str, question=question_text)

        t1 = time.monotonic()
        try:
            answer = call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=512,
                api_key=self.api_key,
            )
        except Exception as e:
            answer = f"[ERROR: {e}]"
        generation_ms = (time.monotonic() - t1) * 1000

        tokens = len(prompt) // 4 + len(answer) // 4
        self._cost.query_tokens += tokens
        self._cost.total_tokens += tokens
        self._cost.total_api_calls += 2  # 1 embedding (search) + 1 LLM (answer)

        return QueryResult(
            answer=answer,
            retrieved_context=context_texts,
            tokens_used=tokens,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

    def reset(self) -> None:
        if self._memory:
            try:
                self._memory.reset()
            except Exception:
                pass
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._cost = CostSnapshot()
        self._init_memory()

    def get_cost(self) -> CostSnapshot:
        return self._cost

    def close(self) -> None:
        """Clean up resources."""
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
