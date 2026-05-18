#!/usr/bin/env python3
"""Helper: produce a golden YAML stub by running the simulator on a source file.

Usage:
    python scripts/gen_golden.py path/to/source.lisp [--stdin "text"] \
        [--tick-limit N] [--log-head 8] [--log-tail 4] [--listing-head 20] \
        [--no-log] > golden_case.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def _str_representer(dumper: yaml.SafeDumper, data: str) -> Any:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_representer, Dumper=yaml.SafeDumper)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csa_lab4.machine import simulate  # noqa: E402
from csa_lab4.translator import translate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--stdin", default="")
    parser.add_argument("--tick-limit", type=int, default=1_000_000)
    parser.add_argument("--log-head", type=int, default=8)
    parser.add_argument("--log-tail", type=int, default=4)
    parser.add_argument("--listing-head", type=int, default=20)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    binary, listing = translate(text)
    stdout, log = simulate(binary, args.stdin, args.tick_limit, log_enabled=not args.no_log)

    case: dict[str, object] = {
        "description": args.description or args.source.stem,
        "source": text,
        "stdin": args.stdin,
        "expected_stdout": stdout,
        "tick_limit": args.tick_limit,
        "log_enabled": not args.no_log,
    }
    if not args.no_log and log:
        lines = log.splitlines()
        case["log_head_lines"] = args.log_head
        case["expected_log_head"] = "\n".join(lines[: args.log_head]) + "\n"
        case["log_tail_lines"] = args.log_tail
        case["expected_log_tail"] = "\n".join(lines[-args.log_tail :]) + "\n"
    if args.listing_head > 0:
        listing_lines = listing.splitlines()
        case["listing_head_lines"] = args.listing_head
        case["expected_listing_head"] = "\n".join(listing_lines[: args.listing_head]) + "\n"

    yaml.safe_dump(case, sys.stdout, sort_keys=False, allow_unicode=True, width=200)
    return 0


if __name__ == "__main__":
    sys.exit(main())
