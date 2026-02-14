#!/usr/bin/env python3
"""
MemoryStress Dataset Generator.

Generates a longitudinal benchmark dataset with facts planted across sessions
over simulated months. Four-stage pipeline:
  1. Facts — 600 facts across 6 categories
  2. Sessions — 1000 sessions across 3 phases
  3. Questions — 300 questions at 4 checkpoints
  4. Validation — referential integrity, coverage, difficulty distribution

Usage:
    python scripts/generate.py --model gpt-4o --seed 42 --output data/memorystress_v1.json --validate
    python scripts/generate.py --small --output data/test.json  # Small test dataset
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from memorystress.llm import call_llm, resolve_provider
from memorystress.schema import (
    BenchmarkData,
    ContradictionChain,
    Fact,
    Question,
    Session,
)
from memorystress.dataset import DatasetLoader

# ── Fact categories ──────────────────────────────────────────────────────────

CATEGORIES = [
    "preference",
    "decision",
    "technical_fact",
    "personal_info",
    "event",
    "relationship",
]

CATEGORY_PROMPTS = {
    "preference": "Generate {n} distinct personal preferences a user might share in conversation (favorite language, food, music genre, workflow habit, tool choice, etc). Output as a JSON array of strings.",
    "decision": "Generate {n} distinct technical or life decisions a user might share (chose framework X, switched to Y, decided to Z). Output as a JSON array of strings.",
    "technical_fact": "Generate {n} distinct technical facts a user might mention in conversation (about programming, tools, systems, APIs). Output as a JSON array of strings.",
    "personal_info": "Generate {n} distinct personal info items (timezone, birthday, pet name, job title, city, family members). Output as a JSON array of strings.",
    "event": "Generate {n} distinct life events a user might mention (conference attended, project launched, moved cities, got promoted). Output as a JSON array of strings.",
    "relationship": "Generate {n} distinct relationship facts (colleague named X works on Y, friend Z lives in W, manager's preference). Output as a JSON array of strings.",
}

# ── Contradiction patterns ───────────────────────────────────────────────────

CONTRADICTION_PATTERNS = [
    "fact_update",           # value changes: A → B
    "fact_update_revert",    # value changes then reverts: A → B → A
    "fact_accumulate",       # value grows: 3 → 5 → 8
    "fact_partial_update",   # part of a fact changes
]

# ── Session generation prompts ───────────────────────────────────────────────

_SESSION_PROMPT = """\
Generate a natural conversation between a user and an AI assistant. The conversation \
should be about "{topic}" and must naturally include the following facts (plant them \
naturally, don't list them):

Facts to plant:
{facts}

The conversation should be {length} turns long (alternating user/assistant). \
Make it feel natural — the facts should emerge organically from the conversation, \
not be stated artificially. {noise_instruction}

Output as a JSON array of objects with "role" (user/assistant) and "content" fields."""

_NOISE_INSTRUCTIONS = {
    "low": "Keep the conversation focused and on-topic.",
    "medium": "Include some tangential discussion and small talk alongside the key facts.",
    "high": "Include substantial off-topic discussion, tangents, and digressions. The key facts should be buried in noise.",
}

# ── Question generation prompt ───────────────────────────────────────────────

_QUESTION_PROMPT = """\
Generate a {qtype} question that tests whether a memory system can recall \
the following fact after it was mentioned {age_desc}:

Fact: {fact_content}
Category: {fact_category}
Context: This fact was planted in session {session_id} on {date}.

Output a JSON object with:
- "question": the natural-language question
- "answer": the correct answer (concise)
- "answer_detail": additional context for the grader
- "difficulty": "easy", "medium", or "hard"

The question should NOT directly quote the fact. Ask about it indirectly."""

# ── Session topic pool ───────────────────────────────────────────────────────

def _extract_json(text: str):
    """Extract JSON from LLM response, stripping markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return json.loads(text)


TOPICS = [
    "setting up a new development environment",
    "debugging a production issue",
    "planning a weekend trip",
    "reviewing a code pull request",
    "discussing project architecture",
    "learning a new programming language",
    "home office setup optimization",
    "career planning and goals",
    "cooking and meal planning",
    "book recommendations and reading habits",
    "fitness routine and health goals",
    "managing a team project",
    "cloud infrastructure migration",
    "personal finance and budgeting",
    "hobby projects and side hustles",
    "conference preparation and talks",
    "database performance tuning",
    "API design and documentation",
    "home renovation planning",
    "travel experiences and tips",
]


def generate_facts(
    n: int, model: str, api_key: str | None, rng: random.Random,
    use_llm: bool = True,
) -> list[Fact]:
    """Stage 1: Generate facts via LLM across categories."""
    facts = []
    per_category = n // len(CATEGORIES)
    remainder = n - per_category * len(CATEGORIES)

    for i, category in enumerate(CATEGORIES):
        count = per_category + (1 if i < remainder else 0)

        if use_llm:
            prompt = CATEGORY_PROMPTS[category].format(n=count)
            try:
                response = call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=4096,
                    temperature=0.8,
                    api_key=api_key,
                )
                items = _extract_json(response)
                if not isinstance(items, list):
                    items = [items]
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning: LLM generation failed for {category}: {e}")
                items = [f"User mentioned something about {category} #{j}" for j in range(count)]
        else:
            items = [f"User mentioned something about {category} #{j}" for j in range(count)]

        for j, content in enumerate(items[:count]):
            fact_id = f"F{len(facts) + 1:03d}"
            importance = rng.randint(1, 5)
            mention_count = rng.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 15, 5])[0]
            facts.append(Fact(
                fact_id=fact_id,
                content=str(content),
                category=category,
                phase_planted=0,
                session_planted="",
                simulated_date="",
                importance=importance,
                mention_count=mention_count,
                difficulty=rng.choice(["easy", "medium", "hard"]),
            ))

    return facts


def assign_facts_to_phases(
    facts: list[Fact],
    n_sessions: int,
    rng: random.Random,
) -> None:
    """Assign each fact to a phase and target session index."""
    phase_boundaries = {
        1: (0, n_sessions // 10),
        2: (n_sessions // 10, n_sessions // 2),
        3: (n_sessions // 2, n_sessions),
    }

    weights = {1: 0.20, 2: 0.35, 3: 0.45}
    for fact in facts:
        phase = rng.choices([1, 2, 3], weights=[weights[1], weights[2], weights[3]])[0]
        lo, hi = phase_boundaries[phase]
        session_idx = rng.randint(lo, max(lo, hi - 1))
        fact.phase_planted = phase
        fact.session_planted = f"S{session_idx + 1:04d}"


def generate_contradiction_chains(
    facts: list[Fact],
    n_chains: int,
    rng: random.Random,
) -> list[ContradictionChain]:
    """Select facts for contradiction chains and generate version sequences."""
    candidates = [
        f for f in facts
        if f.category in ("preference", "decision", "personal_info")
        and f.contradiction_chain is None
    ]
    rng.shuffle(candidates)
    chains = []

    for i, fact in enumerate(candidates[:n_chains]):
        chain_id = f"C{i + 1:03d}"
        pattern = rng.choice(CONTRADICTION_PATTERNS)
        fact.contradiction_chain = chain_id

        versions = [{"version": 1, "content": fact.content, "session_id": fact.session_planted, "phase": fact.phase_planted}]

        if pattern == "fact_update":
            versions.append({"version": 2, "content": f"[Updated] {fact.content}", "session_id": "", "phase": 0})
            final = versions[-1]["content"]
        elif pattern == "fact_update_revert":
            versions.append({"version": 2, "content": f"[Changed] {fact.content}", "session_id": "", "phase": 0})
            versions.append({"version": 3, "content": fact.content, "session_id": "", "phase": 0})
            final = fact.content
        elif pattern == "fact_accumulate":
            versions.append({"version": 2, "content": f"[Accumulated] {fact.content} (more)", "session_id": "", "phase": 0})
            final = versions[-1]["content"]
        else:
            versions.append({"version": 2, "content": f"[Partial update] {fact.content}", "session_id": "", "phase": 0})
            final = versions[-1]["content"]

        chains.append(ContradictionChain(
            chain_id=chain_id,
            original_fact_id=fact.fact_id,
            versions=versions,
            final_ground_truth=final,
            pattern=pattern,
        ))

    return chains


def generate_sessions(
    facts: list[Fact],
    n_sessions: int,
    model: str,
    api_key: str | None,
    rng: random.Random,
    use_llm: bool = True,
) -> list[Session]:
    """Stage 2: Generate sessions with facts planted naturally."""
    session_facts: dict[str, list[Fact]] = {}
    for f in facts:
        sid = f.session_planted
        if sid:
            session_facts.setdefault(sid, []).append(f)

    start_date = datetime(2024, 1, 1)
    sessions = []

    for idx in range(n_sessions):
        session_id = f"S{idx + 1:04d}"
        sim_date = start_date + timedelta(days=idx * 0.3)

        if idx < n_sessions // 10:
            phase = 1
            noise_level = "low"
            n_turns = rng.randint(4, 8)
        elif idx < n_sessions // 2:
            phase = 2
            noise_level = rng.choice(["low", "medium"])
            n_turns = rng.randint(6, 12)
        else:
            phase = 3
            noise_level = rng.choice(["medium", "high"])
            n_turns = rng.randint(8, 16)

        agent_id = rng.choice(["agent_a", "agent_b"])
        topic = rng.choice(TOPICS)

        planted_facts = session_facts.get(session_id, [])
        planted_fact_ids = [f.fact_id for f in planted_facts]

        for f in planted_facts:
            f.simulated_date = sim_date.isoformat()
            if session_id not in f.mention_sessions:
                f.mention_sessions.append(session_id)

        if use_llm and planted_facts:
            facts_text = "\n".join(f"- {f.content}" for f in planted_facts)
            prompt_text = _SESSION_PROMPT.format(
                topic=topic,
                facts=facts_text,
                length=n_turns,
                noise_instruction=_NOISE_INSTRUCTIONS[noise_level],
            )
            try:
                response = call_llm(
                    messages=[{"role": "user", "content": prompt_text}],
                    model=model,
                    max_tokens=4096,
                    temperature=0.7,
                    api_key=api_key,
                )
                turns = _extract_json(response)
                if not isinstance(turns, list):
                    turns = []
            except (json.JSONDecodeError, Exception):
                turns = _make_synthetic_turns(topic, planted_facts, n_turns, rng)
        else:
            turns = _make_synthetic_turns(topic, planted_facts, n_turns, rng)

        token_est = sum(len(t.get("content", "")) for t in turns) // 4

        sessions.append(Session(
            session_id=session_id,
            session_index=idx,
            phase=phase,
            simulated_date=sim_date.isoformat(),
            topic=topic,
            agent_id=agent_id,
            turns=turns,
            planted_fact_ids=planted_fact_ids,
            noise_level=noise_level,
            token_count_estimate=token_est,
        ))

        if (idx + 1) % 100 == 0:
            print(f"  Generated {idx + 1}/{n_sessions} sessions")

    return sessions


def _make_synthetic_turns(
    topic: str, facts: list[Fact], n_turns: int, rng: random.Random
) -> list[dict]:
    """Create synthetic conversation turns without LLM."""
    turns = []
    fact_idx = 0
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "user" and fact_idx < len(facts):
            content = f"By the way, {facts[fact_idx].content}"
            fact_idx += 1
        elif role == "user":
            content = f"Let's continue talking about {topic}."
        else:
            content = f"Sure, that's interesting. Let me help you with {topic}."
        turns.append({"role": role, "content": content})
    return turns


def generate_questions(
    facts: list[Fact],
    chains: list[ContradictionChain],
    sessions: list[Session],
    n_questions: int,
    model: str,
    api_key: str | None,
    rng: random.Random,
    use_llm: bool = True,
) -> list[Question]:
    """Stage 3: Generate questions at 4 phase checkpoints."""
    questions = []
    n_sessions = len(sessions)

    checkpoint_dist = {
        1: int(n_questions * 0.17),
        2: int(n_questions * 0.33),
        3: int(n_questions * 0.33),
        4: n_questions - int(n_questions * 0.17) - int(n_questions * 0.33) * 2,
    }

    type_weights = {
        1: {"fact_recall": 40, "preference_recall": 30, "temporal_ordering": 20, "single_mention_recall": 10},
        2: {"fact_recall": 25, "preference_recall": 15, "contradiction_resolution": 20, "temporal_ordering": 20, "cross_agent_recall": 10, "single_mention_recall": 10},
        3: {"fact_recall": 15, "contradiction_resolution": 20, "temporal_ordering": 15, "cross_agent_recall": 20, "cold_start_recall": 15, "single_mention_recall": 15},
        4: {"single_mention_recall": 40, "contradiction_resolution": 30, "fact_recall": 30},
    }

    fact_by_id = {f.fact_id: f for f in facts}
    chain_by_id = {c.chain_id: c for c in chains}

    for phase_asked in sorted(checkpoint_dist):
        count = checkpoint_dist[phase_asked]
        types = type_weights[phase_asked]
        type_names = list(types.keys())
        type_ws = list(types.values())

        if phase_asked <= 3:
            available_facts = [f for f in facts if f.phase_planted <= phase_asked]
        else:
            available_facts = [
                f for f in facts
                if f.importance <= 2 or f.mention_count == 1
            ]
            if not available_facts:
                available_facts = facts

        for i in range(count):
            qtype = rng.choices(type_names, weights=type_ws, k=1)[0]
            target_fact = rng.choice(available_facts)

            target_session_idx = int(target_fact.session_planted[1:]) - 1 if target_fact.session_planted else 0
            phase_session_boundary = {1: n_sessions // 10, 2: n_sessions // 2, 3: n_sessions, 4: n_sessions}
            current_session_idx = phase_session_boundary.get(phase_asked, n_sessions)
            age_sessions = max(0, current_session_idx - target_session_idx)
            age_days = int(age_sessions * 0.3)

            target_chain_id = None
            cold_start = False
            agent_scope = "agent_a"

            if qtype == "contradiction_resolution" and target_fact.contradiction_chain:
                target_chain_id = target_fact.contradiction_chain
            elif qtype == "contradiction_resolution":
                chain_facts = [f for f in available_facts if f.contradiction_chain]
                if chain_facts:
                    target_fact = rng.choice(chain_facts)
                    target_chain_id = target_fact.contradiction_chain

            if qtype == "cross_agent_recall":
                agent_scope = "agent_b"

            if qtype == "cold_start_recall":
                cold_start = True

            if use_llm:
                age_desc = f"{age_sessions} sessions ago ({age_days} simulated days)"
                q_prompt = _QUESTION_PROMPT.format(
                    qtype=qtype.replace("_", " "),
                    age_desc=age_desc,
                    fact_content=target_fact.content,
                    fact_category=target_fact.category,
                    session_id=target_fact.session_planted,
                    date=target_fact.simulated_date or "unknown",
                )
                try:
                    response = call_llm(
                        messages=[{"role": "user", "content": q_prompt}],
                        model=model,
                        max_tokens=512,
                        temperature=0.5,
                        api_key=api_key,
                    )
                    q_data = _extract_json(response)
                except (json.JSONDecodeError, Exception):
                    q_data = {}
            else:
                q_data = {}

            question_text = q_data.get("question", f"What do you know about: {target_fact.content[:50]}?")
            answer = q_data.get("answer", target_fact.content)
            answer_detail = q_data.get("answer_detail", f"Source fact: {target_fact.content}")
            difficulty = q_data.get("difficulty", target_fact.difficulty)

            scoring_dims = ["recall_at_age"]
            if qtype == "contradiction_resolution":
                scoring_dims.append("contradiction_resolution")
            if qtype == "cross_agent_recall":
                scoring_dims.append("cross_agent_recall")
            if cold_start:
                scoring_dims.append("cold_start_recovery")
            scoring_dims.append("cost_efficiency")

            questions.append(Question(
                question_id=f"Q{len(questions) + 1:03d}",
                question=question_text,
                target_fact_ids=[target_fact.fact_id],
                target_chain_id=target_chain_id,
                answer=answer,
                answer_detail=answer_detail,
                phase_asked=phase_asked,
                target_fact_phase=target_fact.phase_planted,
                age_sessions=age_sessions,
                age_days=age_days,
                question_type=qtype,
                scoring_dimensions=scoring_dims,
                difficulty=difficulty,
                agent_scope=agent_scope,
                cold_start=cold_start,
            ))

    return questions


def validate_dataset(data: BenchmarkData) -> bool:
    """Stage 4: Validate referential integrity and coverage."""
    import tempfile

    from dataclasses import asdict
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(asdict(data), indent=2, default=str))

    loader = DatasetLoader(tmp)
    loader.load()
    warnings = loader.validate()

    tmp.unlink()

    if warnings:
        print(f"\n  Validation warnings ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"    - {w}")
        if len(warnings) > 20:
            print(f"    ... and {len(warnings) - 20} more")
        return False

    print("  Validation passed: no warnings")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="MemoryStress Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="gpt-4o", help="Model for generation (default: gpt-4o)")
    parser.add_argument("--api-key", default=None, help="API key override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="data/memorystress_v1.json", help="Output JSON path")
    parser.add_argument("--validate", action="store_true", help="Run validation after generation")
    parser.add_argument("--small", action="store_true", help="Small test dataset (50 sessions, 30 facts, 20 questions)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (use synthetic placeholders)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    use_llm = not args.no_llm

    if use_llm:
        try:
            config = resolve_provider(args.model)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        import os
        if not args.api_key and not os.environ.get(config["env_key"]):
            print(f"Error: No API key. Set ${config['env_key']} or pass --api-key.")
            print("  Use --no-llm for synthetic generation without API calls.")
            sys.exit(1)

    if args.small:
        n_facts, n_sessions, n_questions, n_chains = 30, 50, 20, 5
    else:
        n_facts, n_sessions, n_questions, n_chains = 600, 1000, 300, 40

    print(f"MemoryStress Dataset Generator")
    print(f"  Model: {args.model} | Seed: {args.seed} | LLM: {use_llm}")
    print(f"  Target: {n_facts} facts, {n_sessions} sessions, {n_questions} questions, {n_chains} contradiction chains")

    print(f"\n[Stage 1/4] Generating {n_facts} facts...")
    facts = generate_facts(n_facts, args.model, args.api_key, rng, use_llm=use_llm)
    print(f"  Generated {len(facts)} facts across {len(CATEGORIES)} categories")

    assign_facts_to_phases(facts, n_sessions, rng)

    print(f"\n[Stage 1b] Generating {n_chains} contradiction chains...")
    chains = generate_contradiction_chains(facts, n_chains, rng)
    print(f"  Generated {len(chains)} chains: {', '.join(c.pattern for c in chains[:5])}...")

    print(f"\n[Stage 2/4] Generating {n_sessions} sessions...")
    sessions = generate_sessions(facts, n_sessions, args.model, args.api_key, rng, use_llm=use_llm)
    print(f"  Generated {len(sessions)} sessions")

    print(f"\n[Stage 3/4] Generating {n_questions} questions...")
    questions = generate_questions(
        facts, chains, sessions, n_questions, args.model, args.api_key, rng, use_llm=use_llm
    )
    print(f"  Generated {len(questions)} questions")

    for phase in [1, 2, 3]:
        p_sessions = sum(1 for s in sessions if s.phase == phase)
        p_facts = sum(1 for f in facts if f.phase_planted == phase)
        print(f"    Phase {phase}: {p_sessions} sessions, {p_facts} facts")

    for phase in [1, 2, 3, 4]:
        p_questions = sum(1 for q in questions if q.phase_asked == phase)
        print(f"    Checkpoint {phase}: {p_questions} questions")

    data = BenchmarkData(
        version="1.0",
        phases={
            "1": {"session_range": [1, n_sessions // 10], "description": "Foundation — short, clean, low noise"},
            "2": {"session_range": [n_sessions // 10 + 1, n_sessions // 2], "description": "Growth — contradictions, medium noise"},
            "3": {"session_range": [n_sessions // 2 + 1, n_sessions], "description": "Stress — dense, high-entropy, multi-topic"},
        },
        stats={
            "total_facts": len(facts),
            "total_sessions": len(sessions),
            "total_questions": len(questions),
            "total_chains": len(chains),
            "seed": args.seed,
            "model": args.model,
        },
        facts=facts,
        contradiction_chains=chains,
        sessions=sessions,
        questions=questions,
    )

    if args.validate:
        print(f"\n[Stage 4/4] Validating dataset...")
        validate_dataset(data)
    else:
        print(f"\n[Stage 4/4] Skipping validation (use --validate to enable)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    output_path.write_text(json.dumps(asdict(data), indent=2, default=str))
    print(f"\n  Dataset written to: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
