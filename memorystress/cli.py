"""CLI entry point for the memorystress package.

Usage:
    memorystress run --dataset data.json --adapter null --grade
    memorystress generate --model gpt-4o --output data.json
"""

from __future__ import annotations

import sys


def main():
    """Route to the appropriate subcommand."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("memorystress - The first longitudinal benchmark for AI memory systems")
        print()
        print("Usage: memorystress <command> [options]")
        print()
        print("Commands:")
        print("  run         Run benchmark against a memory system adapter")
        print("  generate    Generate a new benchmark dataset")
        print()
        print("Run 'memorystress <command> --help' for command-specific options.")
        print()
        print("Examples:")
        print("  memorystress run --dataset data.json --adapter null --grade")
        print("  memorystress run --dataset data.json --adapter omega --grade --extract-facts")
        print("  memorystress generate --model gpt-4o --seed 42 --output data/dataset.json")
        sys.exit(0)

    command = sys.argv.pop(1)
    sys.argv[0] = f"memorystress {command}"

    if command == "run":
        _run_script("run")
    elif command == "generate":
        _run_script("generate")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: run, generate")
        sys.exit(1)


def _run_script(name: str):
    """Import and run a script from the scripts/ directory."""
    import importlib.util
    from pathlib import Path

    # Look for scripts/ relative to common locations
    candidates = [
        Path.cwd() / "scripts" / f"{name}.py",
        Path(__file__).parent.parent / "scripts" / f"{name}.py",
    ]

    for script_path in candidates:
        if script_path.exists():
            spec = importlib.util.spec_from_file_location(name, str(script_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = mod.main()
            sys.exit(result or 0)

    print(f"Error: scripts/{name}.py not found.")
    print("Run this command from the memorystress repository root directory.")
    sys.exit(1)
