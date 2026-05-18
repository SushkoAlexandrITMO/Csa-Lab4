"""ISA encoding round-trip tests."""

from __future__ import annotations

import pytest

from csa_lab4.isa import HAS_OPERAND, Instr, Opcode, decode, encode, mnemonic


@pytest.mark.parametrize("opcode", list(Opcode))
def test_encode_decode_no_operand_roundtrip(opcode: Opcode) -> None:
    if opcode in HAS_OPERAND:
        return
    instr = Instr(opcode=opcode)
    assert decode(encode(instr)) == instr


@pytest.mark.parametrize("operand", [0, 1, -1, 42, -42, 0x7FFF, -0x8000])
def test_encode_decode_operand_roundtrip(operand: int) -> None:
    instr = Instr(opcode=Opcode.PUSH, operand=operand)
    assert decode(encode(instr)) == instr


def test_mnemonic_with_operand() -> None:
    assert mnemonic(Instr(opcode=Opcode.PUSH, operand=0x42)) == "PUSH 0x0042"


def test_mnemonic_without_operand() -> None:
    assert mnemonic(Instr(opcode=Opcode.ADD)) == "ADD"
