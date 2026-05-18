"""Microcode ROM and signal definitions (stub — populated in phase 3)."""

from __future__ import annotations

from enum import Enum, auto


class Signal(Enum):
    LATCH_AR = auto()
    LATCH_DR = auto()
    LATCH_PC = auto()
    LATCH_M_PC = auto()
    LATCH_IR = auto()
    LATCH_TOS = auto()
    DS_PUSH = auto()
    DS_POP = auto()
    RS_PUSH = auto()
    RS_POP = auto()
    ALU_OP = auto()
    SET_FLAGS = auto()
    MEM_READ = auto()
    MEM_WRITE = auto()
    IO_READ = auto()
    IO_WRITE = auto()
    HALT = auto()


class Sel(Enum):
    PC = auto()
    TOS = auto()
    NOS = auto()
    DR = auto()
    IR_OPERAND = auto()
    ALU = auto()
    PLUS_ONE = auto()
    ZERO = auto()
    OPCODE_DISPATCH = auto()


class AluOp(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    PASS_LEFT = auto()
