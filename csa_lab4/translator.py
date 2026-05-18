"""Lisp -> binary translator (stub — populated in phase 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def translate(source: str) -> tuple[bytes, str]:
    """Translate Lisp source to (binary blob, debug listing). Stub."""
    raise NotImplementedError("translator implementation pending")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="translator", description="Lisp -> binary translator")
    parser.add_argument("source", type=Path, help="Lisp source file")
    parser.add_argument("output", type=Path, help="Output binary file (.bin)")
    parser.add_argument("--listing", type=Path, default=None, help="Optional debug listing (.lst)")
    args = parser.parse_args(argv)

    source_text = args.source.read_text(encoding="utf-8")
    binary, listing = translate(source_text)
    args.output.write_bytes(binary)
    if args.listing is not None:
        args.listing.write_text(listing, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
