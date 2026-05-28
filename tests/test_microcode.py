"""Structural tests for the microprogram ROM."""

from __future__ import annotations

from csa_lab4.isa import Opcode
from csa_lab4.microcode import DISPATCH, M_FETCH, MPROGRAM, MicroInstr, Sel, Signal


def test_dispatch_covers_all_opcodes() -> None:
    missing = set(Opcode) - DISPATCH.keys()
    assert not missing, f"opcodes without microcode entry: {missing}"


def test_dispatch_addresses_within_rom() -> None:
    for opcode, address in DISPATCH.items():
        assert 0 <= address < len(MPROGRAM), f"dispatch[{opcode}]={address} out of range"


def test_fetch_address_is_zero() -> None:
    assert M_FETCH == 0


def _terminates(mi: MicroInstr) -> bool:
    return any(sig is Signal.LATCH_M_PC or sig is Signal.HALT for sig, _ in mi.signals)


def test_every_microinstruction_advances_m_pc() -> None:
    for index, mi in enumerate(MPROGRAM):
        assert _terminates(mi), f"microinstruction {index} ({mi.label}) does not advance m_PC"


def test_signal_selectors_are_consistent() -> None:
    no_sel_signals = {
        Signal.LATCH_IR,
        Signal.LATCH_FLAGS,
        Signal.LATCH_CARRY,
        Signal.MEM_READ,
        Signal.DS_POP,
        Signal.RS_POP,
        Signal.HALT,
    }
    for index, mi in enumerate(MPROGRAM):
        for signal, selector in mi.signals:
            if signal in no_sel_signals:
                assert selector is None, f"signal {signal.name} in mi {index} must not carry a selector"
            else:
                assert selector is not None, f"signal {signal.name} in mi {index} requires a selector"
                assert isinstance(selector, Sel)


def test_handler_endpoints_return_to_fetch_or_halt() -> None:
    """Every executor (entry reached via DISPATCH) must reach FETCH or HALT in finite steps."""
    for opcode, start in DISPATCH.items():
        visited: set[int] = set()
        cursor = start
        while cursor not in visited:
            visited.add(cursor)
            mi = MPROGRAM[cursor]
            terminators = [(s, sel) for s, sel in mi.signals if s is Signal.LATCH_M_PC]
            halted = any(s is Signal.HALT for s, _ in mi.signals)
            if halted:
                break
            assert terminators, f"opcode {opcode}: mi {cursor} has no LATCH_M_PC"
            assert len(terminators) == 1, f"opcode {opcode}: mi {cursor} has multiple LATCH_M_PC"
            _, target_sel = terminators[0]
            if target_sel is Sel.M_PC_FETCH:
                break
            if target_sel is Sel.M_PC_PLUS_ONE:
                cursor += 1
                continue
            if target_sel is Sel.M_PC_DISPATCH:
                break  # dispatch is data-dependent; covered separately by test_dispatch_*
            raise AssertionError(f"unexpected M_PC selector {target_sel}")
        else:
            raise AssertionError(f"opcode {opcode}: microcode loop without termination")
