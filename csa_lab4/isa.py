"""Instruction Set Architecture: opcodes, encoding, decoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Opcode(IntEnum):
    NOP = 0x00
    HALT = 0x01

    JMP = 0x02
    JZ = 0x03
    JNZ = 0x04
    JS = 0x05
    CALL = 0x06
    RET = 0x07

    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14
    CMP = 0x15
    NEG = 0x16
    INC = 0x17
    DEC = 0x18

    LOAD = 0x20
    STORE = 0x21
    LOADI = 0x22
    STOREI = 0x23

    # Address registers A and B with post-increment indirection.
    # A/B are 16-bit pointers used ONLY for memory access — they never
    # participate in ALU operations, so the architecture remains stack-based
    # for computation while pointer iteration becomes a single instruction.
    LDA = 0x24  # A <- operand (immediate address)
    LDB = 0x25  # B <- operand
    LDAI = 0x26  # push MEM[A]; A <- A + 1
    LDBI = 0x27  # push MEM[B]; B <- B + 1
    STAI = 0x28  # MEM[A] <- pop; A <- A + 1
    STBI = 0x29  # MEM[B] <- pop; B <- B + 1

    INPUT = 0x30
    OUTPUT = 0x31

    PUSH = 0x40
    DUP = 0x50
    DROP = 0x51
    SWAP = 0x52
    OVER = 0x53


HAS_OPERAND: frozenset[Opcode] = frozenset(
    {
        Opcode.JMP,
        Opcode.JZ,
        Opcode.JNZ,
        Opcode.JS,
        Opcode.CALL,
        Opcode.LOAD,
        Opcode.STORE,
        Opcode.INPUT,
        Opcode.OUTPUT,
        Opcode.PUSH,
        Opcode.LDA,
        Opcode.LDB,
    },
)

WORD_BITS: int = 32
WORD_MASK: int = (1 << WORD_BITS) - 1
OPERAND_BITS: int = 16
OPERAND_MASK: int = (1 << OPERAND_BITS) - 1
OPERAND_SIGN_BIT: int = 1 << (OPERAND_BITS - 1)


@dataclass(frozen=True)
class Instr:
    opcode: Opcode
    operand: int = 0

    def has_operand(self) -> bool:
        return self.opcode in HAS_OPERAND


def encode(instr: Instr) -> int:
    """Pack instruction as a 32-bit word: [opcode:8][reserved:8][operand:16]."""
    operand = instr.operand & OPERAND_MASK if instr.has_operand() else 0
    return ((int(instr.opcode) & 0xFF) << 24) | (operand & OPERAND_MASK)


def decode(word: int) -> Instr:
    raw_opcode = (word >> 24) & 0xFF
    opcode = Opcode(raw_opcode)
    raw_operand = word & OPERAND_MASK
    if opcode in HAS_OPERAND and raw_operand & OPERAND_SIGN_BIT:
        operand = raw_operand - (1 << OPERAND_BITS)
    else:
        operand = raw_operand
    return Instr(opcode=opcode, operand=operand)


def mnemonic(instr: Instr) -> str:
    if instr.has_operand():
        return f"{instr.opcode.name} 0x{instr.operand & OPERAND_MASK:04X}"
    return instr.opcode.name
