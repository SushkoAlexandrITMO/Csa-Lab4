"""Tick-accurate microcoded stack machine simulator (stub — populated in phase 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def simulate(binary: bytes, stdin_data: str, tick_limit: int) -> tuple[str, str]:
    """Run the binary against stdin_data. Returns (stdout, log). Stub."""
    raise NotImplementedError("machine implementation pending")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="machine", description="Stack machine simulator")
    parser.add_argument("binary", type=Path, help="Compiled binary file (.bin)")
    parser.add_argument("stdin", type=Path, help="Input stream file")
    parser.add_argument("--log", type=Path, default=None, help="Optional log file")
    parser.add_argument("--tick-limit", type=int, default=1_000_000, help="Max ticks to simulate")
    args = parser.parse_args(argv)

    binary = args.binary.read_bytes()
    stdin_data = args.stdin.read_text(encoding="utf-8") if args.stdin.exists() else ""
    stdout, log_text = simulate(binary, stdin_data, args.tick_limit)
    sys.stdout.write(stdout)
    if args.log is not None:
        args.log.write_text(log_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
