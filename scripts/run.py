#!/usr/bin/env python3
"""
MemoryStress Benchmark Harness.

Runs a memory system adapter against the MemoryStress dataset, measuring
degradation, cost, and recall across simulated months of usage.

Usage:
    python scripts/run.py --dataset data/memorystress_v1.json --adapter omega --grade
    python scripts/run.py --dataset data/memorystress_v1.json --adapter null --grade
    python scripts/run.py --dataset data/memorystress_v1.json --adapter mem0 --grade
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from memorystress.cost_tracker import CostTracker
from memorystress.dataset import DatasetLoader
from memorystress.grader import grade_answer
from memorystress.llm import resolve_provider
from memorystress.metrics import GradedQuestion, MetricsEngine
from memorystress.report import generate_report


def create_adapter(name: str, model: str, api_key: str | None,
                   extract_facts: bool = False, query_augment: bool = False):
    """Factory for memory system adapters."""
    if name == "omega":
        from memorystress.adapters.omega_adapter import OmegaAdapter
        return OmegaAdapter(model=model, api_key=api_key,
                            extract_facts=extract_facts,
                            query_augment=query_augment)
    elif name == "mem0":
        from memorystress.adapters.mem0_adapter import Mem0Adapter
        return Mem0Adapter(model=model, api_key=api_key)
    elif name == "raw_context":
        from memorystress.adapters.raw_context_adapter import RawContextAdapter
        return RawContextAdapter(model=model, api_key=api_key)
    elif name == "null":
        from memorystress.adapters.null_adapter import NullAdapter
        return NullAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name}. Choose: omega, mem0, raw_context, null")


def run_harness(args) -> int:
    """Main benchmark execution."""
    print(f"Loading dataset: {args.dataset}")
    loader = DatasetLoader(args.dataset)
    data = loader.load()
    print(f"  {len(data.sessions)} sessions, {len(data.questions)} questions, {len(data.facts)} facts")

    warnings = loader.validate()
    if warnings:
        print(f"  Dataset warnings: {len(warnings)}")
        for w in warnings[:5]:
            print(f"    - {w}")

    print(f"\nAdapter: {args.adapter} | Model: {args.model}")
    adapter = create_adapter(args.adapter, args.model, args.api_key,
                             extract_facts=getattr(args, "extract_facts", False),
                             query_augment=getattr(args, "query_augment", False))
    adapter.reset()

    cost_tracker = CostTracker(model=args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hypothesis_path = output_dir / "hypotheses.jsonl"
    completed_ids: set[str] = set()
    hypotheses: list[dict] = []

    if args.resume and hypothesis_path.exists():
        with open(hypothesis_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    completed_ids.add(entry["question_id"])
                    hypotheses.append(entry)
        print(f"  Resuming: {len(completed_ids)} questions already completed")

    start_time = time.time()

    # Phase 1-3: Ingest + Checkpoint
    for phase in [1, 2, 3]:
        phase_sessions = loader.sessions_for_phase(phase)
        phase_questions = loader.questions_for_phase(phase)

        print(f"\n{'='*60}")
        print(f"  Phase {phase}: {len(phase_sessions)} sessions, {len(phase_questions)} questions")
        print(f"{'='*60}")

        print(f"  Ingesting sessions...")
        for i, session in enumerate(phase_sessions):
            session_dict = asdict(session)
            result = adapter.ingest(session_dict)
            if result.tokens_used:
                cost_tracker.record_ingest(phase, result.tokens_used)

            if (i + 1) % 50 == 0 or i == len(phase_sessions) - 1:
                print(f"    [{i + 1}/{len(phase_sessions)}] ingested")

        if phase_questions:
            print(f"  Running {len(phase_questions)} checkpoint questions...")
            hyp_file = open(hypothesis_path, "a")

            for i, question in enumerate(phase_questions):
                qid = question.question_id
                if qid in completed_ids:
                    continue

                question_dict = asdict(question)
                adapter.set_agent_id(question.agent_scope)
                qresult = adapter.query(question_dict)
                cost_tracker.record_query(phase, qresult.tokens_used)

                entry = {
                    "question_id": qid,
                    "phase_asked": phase,
                    "question_type": question.question_type,
                    "hypothesis": qresult.answer,
                    "retrieval_latency_ms": qresult.retrieval_latency_ms,
                    "generation_latency_ms": qresult.generation_latency_ms,
                }
                hypotheses.append(entry)
                completed_ids.add(qid)

                hyp_file.write(json.dumps(entry) + "\n")
                hyp_file.flush()

                if args.verbose:
                    print(f"    [{i + 1}/{len(phase_questions)}] {qid} ({question.question_type})")
                    print(f"      Q: {question.question[:80]}")
                    print(f"      A: {qresult.answer[:80]}")
                elif (i + 1) % 10 == 0:
                    print(f"    [{i + 1}/{len(phase_questions)}] answered")

            hyp_file.close()

    # Phase 4: Recovery (no new sessions)
    phase4_questions = loader.questions_for_phase(4)
    if phase4_questions:
        print(f"\n{'='*60}")
        print(f"  Phase 4 (Recovery): {len(phase4_questions)} questions (no new sessions)")
        print(f"{'='*60}")

        hyp_file = open(hypothesis_path, "a")
        for i, question in enumerate(phase4_questions):
            qid = question.question_id
            if qid in completed_ids:
                continue

            question_dict = asdict(question)
            adapter.set_agent_id(question.agent_scope)
            qresult = adapter.query(question_dict)
            cost_tracker.record_query(4, qresult.tokens_used)

            entry = {
                "question_id": qid,
                "phase_asked": 4,
                "question_type": question.question_type,
                "hypothesis": qresult.answer,
                "retrieval_latency_ms": qresult.retrieval_latency_ms,
                "generation_latency_ms": qresult.generation_latency_ms,
            }
            hypotheses.append(entry)
            completed_ids.add(qid)
            hyp_file.write(json.dumps(entry) + "\n")
            hyp_file.flush()

        hyp_file.close()
        print(f"  {len(phase4_questions)} recovery questions answered")

    # Cold Start
    cold_questions = loader.cold_start_questions()
    cold_not_done = [q for q in cold_questions if q.question_id not in completed_ids]
    if cold_not_done:
        print(f"\n{'='*60}")
        print(f"  Cold Start: {len(cold_not_done)} questions")
        print(f"{'='*60}")

        hyp_file = open(hypothesis_path, "a")
        for question in cold_not_done:
            qid = question.question_id
            question_dict = asdict(question)
            adapter.set_agent_id(question.agent_scope)
            qresult = adapter.query(question_dict)
            cost_tracker.record_query(question.phase_asked, qresult.tokens_used)

            entry = {
                "question_id": qid,
                "phase_asked": question.phase_asked,
                "question_type": question.question_type,
                "hypothesis": qresult.answer,
                "cold_start": True,
                "retrieval_latency_ms": qresult.retrieval_latency_ms,
                "generation_latency_ms": qresult.generation_latency_ms,
            }
            hypotheses.append(entry)
            completed_ids.add(qid)
            hyp_file.write(json.dumps(entry) + "\n")
            hyp_file.flush()

        hyp_file.close()

    elapsed = time.time() - start_time
    print(f"\n  Generation complete: {len(hypotheses)} hypotheses in {elapsed / 60:.1f} min")

    # Grading
    graded_questions: list[GradedQuestion] = []

    if args.grade:
        print(f"\nGrading {len(hypotheses)} hypotheses...")
        question_by_id = {q.question_id: q for q in data.questions}

        for i, hyp in enumerate(hypotheses):
            qid = hyp["question_id"]
            q = question_by_id.get(qid)
            if not q:
                continue

            question_dict = asdict(q)
            is_correct, grade_tokens = grade_answer(
                question_dict, hyp["hypothesis"],
                model=args.model, api_key=args.api_key,
            )
            cost_tracker.record_grading(hyp.get("phase_asked", 0), grade_tokens)
            cost_tracker.record_result(hyp.get("phase_asked", 0), is_correct)

            graded_questions.append(GradedQuestion(
                question_id=qid,
                question_type=q.question_type,
                phase_asked=q.phase_asked,
                age_sessions=q.age_sessions,
                correct=is_correct,
                hypothesis=hyp["hypothesis"],
                cold_start=hyp.get("cold_start", False),
            ))

            if args.verbose:
                status = "PASS" if is_correct else "FAIL"
                print(f"  [{i + 1}/{len(hypotheses)}] [{status}] {qid} ({q.question_type})")
            elif (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(hypotheses)}] graded")

        graded_path = output_dir / "graded.jsonl"
        with open(graded_path, "w") as f:
            for gq in graded_questions:
                f.write(json.dumps(asdict(gq)) + "\n")
        print(f"  Grading results: {graded_path}")

    # Metrics
    if graded_questions:
        print(f"\nComputing metrics...")
        engine = MetricsEngine(cost_tracker=cost_tracker)
        metrics = engine.compute(graded_questions)

        print(f"\n{'='*60}")
        print(f"  MemoryStress Benchmark Results ({args.adapter})")
        print(f"{'='*60}")
        print(f"  Overall: {metrics.total_correct}/{metrics.total_questions} ({metrics.overall_accuracy:.1%})")
        print(f"  Degradation slope: {metrics.degradation_slope:+.4f}")
        print()

        for phase in sorted(metrics.degradation_curve):
            acc = metrics.degradation_curve[phase]
            bar = "#" * int(acc * 20)
            print(f"  Phase {phase}: {acc:.1%} {bar}")

        print(f"\n  Contradiction: {metrics.contradiction_accuracy:.1%} ({metrics.contradiction_total} Qs)")
        print(f"  Cross-agent:   {metrics.cross_agent_accuracy:.1%} ({metrics.cross_agent_total} Qs)")
        print(f"  Cold start:    {metrics.cold_start_accuracy:.1%} ({metrics.cold_start_total} Qs)")

        for bucket, acc in sorted(metrics.recall_at_age.items()):
            print(f"  Recall@{bucket}: {acc:.1%}")

        report_path = generate_report(
            metrics, cost_tracker, args.adapter, args.model, output_dir
        )
        print(f"\n  Report: {report_path}")
        print(f"  Metrics: {output_dir / 'metrics.json'}")
        print(f"  Degradation: {output_dir / 'degradation_curve.json'}")

    if hasattr(adapter, "close"):
        adapter.close()

    total_elapsed = time.time() - start_time
    print(f"\n  Total time: {total_elapsed / 60:.1f} minutes")
    cost_snap = cost_tracker.get_total_snapshot()
    print(f"  Estimated cost: ${cost_snap.estimated_cost_usd:.2f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="MemoryStress Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --dataset data.json --adapter null --grade               # sanity check (~0%%)
  %(prog)s --dataset data.json --adapter omega --grade --verbose    # full OMEGA run
  %(prog)s --dataset data.json --adapter mem0 --grade               # Mem0 run
  %(prog)s --dataset data.json --adapter raw_context --grade        # context window baseline
  %(prog)s --dataset data.json --adapter omega --resume             # resume interrupted run
""",
    )
    parser.add_argument("--dataset", required=True, help="Path to MemoryStress JSON dataset")
    parser.add_argument("--adapter", default="omega", choices=["omega", "mem0", "raw_context", "null"], help="Memory system adapter (default: omega)")
    parser.add_argument("--model", default="gpt-4o", help="Model for generation and grading (default: gpt-4o)")
    parser.add_argument("--api-key", default=None, help="API key override")
    parser.add_argument("--output-dir", default="results/run", help="Output directory for results")
    parser.add_argument("--grade", action="store_true", help="Grade hypotheses after generation")
    parser.add_argument("--resume", action="store_true", help="Resume from existing hypotheses file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-question details")
    parser.add_argument("--extract-facts", action="store_true", help="Extract facts at ingest (OMEGA only, ~$0.50 extra)")
    parser.add_argument("--query-augment", action="store_true", help="LLM query augmentation (OMEGA only, ~$0.30 extra)")
    args = parser.parse_args()

    if args.adapter not in ("null",):
        import os
        try:
            config = resolve_provider(args.model)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if not args.api_key and not os.environ.get(config["env_key"]):
            print(f"Error: No API key. Set ${config['env_key']} or pass --api-key.")
            sys.exit(1)

    return run_harness(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
