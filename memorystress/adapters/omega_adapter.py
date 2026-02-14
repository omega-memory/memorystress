"""OMEGA adapter — uses omega-memory pip package for memory storage and retrieval."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import datetime

from memorystress.adapters.base import (
    CostSnapshot,
    IngestResult,
    MemorySystemAdapter,
    QueryResult,
)
from memorystress.llm import call_llm

try:
    from omega.knowledge.sqlite_store import SQLiteStore
except ImportError:
    SQLiteStore = None

# RAG prompt — contradiction-aware, chronological ordering
_RAG_PROMPT = """\
I will give you several notes from past conversations. \
Please answer the question based on the relevant notes. \
If the question cannot be answered based on the provided notes, say so.

CRITICAL RULES:
1. Notes are sorted from OLDEST to NEWEST. The LAST note is the most recent.
2. When multiple notes discuss the same topic but give DIFFERENT information, \
ALWAYS use the information from the MOST RECENT note (highest note number). \
Earlier values are OUTDATED and SUPERSEDED — ignore them entirely.
3. If the question asks about a preference, opinion, or decision that may have \
changed over time, find the most recent note on that topic and ONLY use that one.
4. If the question asks "how many" or for a count, enumerate all items and state the total.
5. Give a direct, concise answer.

Notes from past conversations (oldest first, newest last):

{notes}

Question: {question}
Answer:"""

# Query augmentation prompt — generates alternative search terms
_QUERY_AUGMENT_PROMPT = """\
Given this question about a user's personal history, generate 3 alternative \
search queries that could find the answer in their conversation logs. Focus on \
SYNONYMS, RELATED TERMS, and KEY NOUNS that the user might have used instead of \
the exact words in the question. Be specific and concrete.

Question: {question}

Output ONLY the alternative search queries, one per line. Keep each under 15 words."""

# Fact extraction prompt for single-mention recall improvement
_FACT_EXTRACTION_PROMPT = """\
Extract ALL discrete facts, preferences, decisions, and personal details from this \
conversation. Include exact dates, names, numbers, preferences, opinions, relationships, \
and any specific information the user shared. Each fact should be a complete, standalone \
statement that can be understood without context.

Conversation:
{content}

Facts (one per line, no numbering, no bullets):"""


def _format_turns(turns: list[dict]) -> str:
    """Format session turns as text for ingestion."""
    lines = []
    for turn in turns:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines)


def _format_note(content: str, date_str: str, index: int) -> str:
    """Format a retrieved memory as a numbered note block."""
    return (
        f"[Note {index} | Date: {date_str}]\n"
        f"{content}\n"
        f"[End Note {index}]"
    )


def _boost_recency(results: list) -> list:
    """Boost relevance scores by recency — newer notes rank higher."""
    dates = []
    for r in results:
        if r.metadata:
            d = r.metadata.get("referenced_date", "")
            if d:
                dates.append(d)
    if not dates:
        return results

    dates.sort()
    try:
        earliest = datetime.fromisoformat(dates[0])
        latest = datetime.fromisoformat(dates[-1])
    except (ValueError, IndexError):
        return results

    span = (latest - earliest).total_seconds()
    if span <= 0:
        return results

    for r in results:
        d = (r.metadata or {}).get("referenced_date", "")
        if d:
            try:
                t = datetime.fromisoformat(d)
                # Recency factor: 1.0 (oldest) to 1.8 (newest)
                frac = (t - earliest).total_seconds() / span
                r.relevance = (r.relevance or 0) * (1.0 + 0.8 * frac)
            except ValueError:
                pass
    return results


def _sort_by_date(results: list) -> list:
    """Sort results chronologically (oldest first) by referenced_date."""
    def date_key(r):
        d = (r.metadata or {}).get("referenced_date", "")
        return d or "0000-00-00"
    return sorted(results, key=date_key)


class OmegaAdapter(MemorySystemAdapter):
    """Adapter wrapping OMEGA's SQLiteStore.

    Requires: pip install omega-memory
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        retrieval_limit: int = 20,
        extract_facts: bool = False,
        query_augment: bool = False,
        fact_model: str = "gpt-4.1-mini",
    ):
        if SQLiteStore is None:
            raise ImportError(
                "OMEGA adapter requires omega-memory. "
                "Install it with: pip install omega-memory"
            )
        self.model = model
        self.api_key = api_key
        self.retrieval_limit = retrieval_limit
        self.extract_facts = extract_facts
        self.query_augment = query_augment
        self.fact_model = fact_model
        self._agent_id: str = "agent_a"
        self._tmpdir: str | None = None
        self._store = None
        self._cost = CostSnapshot()
        self._init_store()

    def _init_store(self) -> None:
        """Create a fresh temp directory and SQLiteStore."""
        self._tmpdir = tempfile.mkdtemp(prefix="memorystress_omega_")
        os.environ["OMEGA_HOME"] = self._tmpdir
        db_path = os.path.join(self._tmpdir, "bench.db")
        self._store = SQLiteStore(db_path=db_path)

    def set_agent_id(self, agent_id: str) -> None:
        self._agent_id = agent_id

    def _augment_query(self, question: str) -> list[str]:
        """Use a cheap LLM to generate alternative search queries."""
        try:
            response = call_llm(
                messages=[{"role": "user", "content": _QUERY_AUGMENT_PROMPT.format(
                    question=question
                )}],
                model=self.fact_model,
                max_tokens=100,
                temperature=0.3,
                api_key=self.api_key,
            )
            queries = [line.strip() for line in response.strip().split("\n") if line.strip()]
            # Track cost
            tokens = len(question) // 4 + 100
            self._cost.query_tokens += tokens
            self._cost.total_tokens += tokens
            self._cost.total_api_calls += 1
            return queries[:3]
        except Exception:
            return []

    def _extract_facts_from_content(self, content: str) -> list[str]:
        """Extract discrete facts from session content using a cheap LLM."""
        try:
            response = call_llm(
                messages=[{"role": "user", "content": _FACT_EXTRACTION_PROMPT.format(
                    content=content[:4000]
                )}],
                model=self.fact_model,
                max_tokens=500,
                temperature=0,
                api_key=self.api_key,
            )
            facts = [
                line.strip().lstrip("- 0123456789.)").strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            return facts
        except Exception:
            return []

    def ingest(self, session: dict) -> IngestResult:
        content = _format_turns(session.get("turns", []))
        if not content.strip():
            return IngestResult(success=True)

        session_id = session.get("session_id", "")
        agent_id = session.get("agent_id", self._agent_id)
        simulated_date = session.get("simulated_date", "")
        phase = session.get("phase", 0)

        try:
            # Store the full session
            self._store.store(
                content=content,
                session_id=session_id,
                metadata={
                    "event_type": "session_summary",
                    "referenced_date": simulated_date,
                    "priority": 3,
                    "agent_type": agent_id,
                    "phase": phase,
                },
                skip_inference=True,
            )
            tokens = len(content) // 4
            api_calls = 1

            # Extract and store discrete facts for better recall
            if self.extract_facts:
                facts = self._extract_facts_from_content(content)
                for fact in facts:
                    self._store.store(
                        content=fact,
                        session_id=f"{session_id}_fact",
                        metadata={
                            "event_type": "extracted_fact",
                            "referenced_date": simulated_date,
                            "priority": 4,
                            "agent_type": agent_id,
                            "phase": phase,
                            "source_session": session_id,
                        },
                        skip_inference=True,
                    )
                fact_tokens = sum(len(f) // 4 for f in facts)
                extract_tokens = len(content[:4000]) // 4 + 500 // 4
                tokens += fact_tokens + extract_tokens
                api_calls += 1

            self._cost.ingest_tokens += tokens
            self._cost.total_tokens += tokens
            self._cost.total_api_calls += api_calls
            return IngestResult(success=True, tokens_used=tokens, api_calls=api_calls)
        except Exception as e:
            return IngestResult(success=False, error=str(e))

    def query(self, question: dict) -> QueryResult:
        question_text = question.get("question", "")
        agent_scope = question.get("agent_scope", self._agent_id)

        t0 = time.monotonic()
        results = self._store.query(
            question_text,
            limit=self.retrieval_limit,
            agent_type=agent_scope,
            include_infrastructure=True,
        )

        # Cross-agent fallback: when agent-scoped query returns too few
        # results, do an unscoped query to catch cross-agent facts
        if len(results) < 5:
            unscoped = self._store.query(
                question_text,
                limit=self.retrieval_limit,
                include_infrastructure=True,
            )
            seen = {r.id for r in results}
            for r in unscoped:
                if r.id not in seen:
                    results.append(r)
                    seen.add(r.id)

        # Query augmentation: generate alternative search queries and
        # merge results for better recall of single-mention facts
        if self.query_augment:
            aug_queries = self._augment_query(question_text)
            seen = {r.id for r in results}
            for aq in aug_queries:
                aug_results = self._store.query(
                    aq,
                    limit=self.retrieval_limit // 2,
                    agent_type=agent_scope,
                    include_infrastructure=True,
                )
                for r in aug_results:
                    if r.id not in seen:
                        results.append(r)
                        seen.add(r.id)

        # Apply recency boosting — newer notes ranked higher
        results = _boost_recency(results)

        # Re-sort by score after recency boost, take top N
        results.sort(key=lambda r: r.relevance or 0, reverse=True)
        results = results[:self.retrieval_limit]

        # Sort chronologically for the RAG prompt (oldest first)
        results = _sort_by_date(results)

        retrieval_ms = (time.monotonic() - t0) * 1000

        # Format retrieved notes
        note_blocks = []
        context_texts = []
        for i, r in enumerate(results, 1):
            date_str = "Unknown"
            if r.metadata:
                date_str = r.metadata.get("referenced_date", "Unknown")
            note_blocks.append(_format_note(r.content, date_str, i))
            context_texts.append(r.content[:200])

        notes_str = "\n\n".join(note_blocks) if note_blocks else "(No relevant notes found)"
        prompt = _RAG_PROMPT.format(notes=notes_str, question=question_text)

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
        self._cost.total_api_calls += 1

        return QueryResult(
            answer=answer,
            retrieved_context=context_texts,
            tokens_used=tokens,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

    def reset(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._cost = CostSnapshot()
        self._init_store()

    def get_cost(self) -> CostSnapshot:
        return self._cost

    def close(self) -> None:
        """Clean up resources."""
        if self._store:
            self._store.close()
            self._store = None
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
