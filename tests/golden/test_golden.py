"""Golden-test runner: every ``*.yml`` next to this file is one integration test.

Each YAML file describes one program end-to-end:

  description     human-readable label
  source          lisp source code
  stdin           input stream fed into port 0
  expected_stdout exact stdout produced by the simulator
  tick_limit      hard cap on simulated ticks
  log_head_lines  number of leading log lines to assert against
  log_tail_lines  number of trailing log lines to assert against
  expected_log_head  text the log must start with
  expected_log_tail  text the log must end with
  expected_listing_head  text the .lst listing must start with

``expected_*`` fields are optional — when absent, only stdout and the tick
budget are checked. ``log_enabled`` defaults to true; set to false for
algorithms whose tick count would make the log gigabytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from csa_lab4.machine import simulate
from csa_lab4.translator import translate

GOLDEN_DIR = Path(__file__).parent


def _discover_cases() -> list[Path]:
    return sorted(GOLDEN_DIR.glob("*.yml"))


@pytest.mark.parametrize("case_path", _discover_cases(), ids=lambda p: p.stem)
def test_golden(case_path: Path) -> None:
    case: dict[str, Any] = yaml.safe_load(case_path.read_text(encoding="utf-8"))

    source: str = case["source"]
    stdin: str = case.get("stdin", "")
    expected_stdout: str = case.get("expected_stdout", "")
    tick_limit: int = int(case.get("tick_limit", 1_000_000))
    log_enabled: bool = bool(case.get("log_enabled", True))

    binary, listing = translate(source)
    stdout, log = simulate(binary, stdin, tick_limit, log_enabled=log_enabled)

    assert stdout == expected_stdout, f"stdout mismatch for {case_path.name}"

    if "expected_log_head" in case:
        head_lines = int(case.get("log_head_lines", 0))
        actual_head = "\n".join(log.splitlines()[:head_lines]) + "\n"
        assert actual_head == case["expected_log_head"], f"log head mismatch in {case_path.name}"

    if "expected_log_tail" in case:
        tail_lines = int(case.get("log_tail_lines", 0))
        actual_tail = "\n".join(log.splitlines()[-tail_lines:]) + "\n"
        assert actual_tail == case["expected_log_tail"], f"log tail mismatch in {case_path.name}"

    if "expected_listing_head" in case:
        head_lines = int(case.get("listing_head_lines", 0))
        actual_head = "\n".join(listing.splitlines()[:head_lines]) + "\n"
        assert actual_head == case["expected_listing_head"], f"listing head mismatch in {case_path.name}"
