"""Microprogram ROM, signals, selectors and the dispatch table.

The microcoded control unit runs at one microinstruction per tick. Each
microinstruction is a fixed set of control signals that fire in parallel.

Tick semantics (synchronous, sample-before-write):
  * all source values (TOS, DS top, AR, DR, PC, flags, IR.operand, memory)
    are sampled at the rising edge that starts the tick;
  * all destinations latch the new values at the rising edge that ends the
    tick. Combinatorial paths (ALU, memory read, dispatch ROM) compute their
    outputs from start-of-tick sources.

The microprogram is a flat ROM. ``DISPATCH`` maps an opcode to the entry
point of the corresponding executor. Each executor ends with a
``LATCH_M_PC -> M_PC_FETCH`` signal returning control to the fetch routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from csa_lab4.isa import Opcode


class Signal(Enum):
    """Discrete control lines."""

    LATCH_PC = auto()
    LATCH_AR = auto()
    LATCH_DR = auto()
    LATCH_IR = auto()
    LATCH_TOS = auto()
    LATCH_FLAGS = auto()
    LATCH_M_PC = auto()

    DS_PUSH = auto()
    DS_POP = auto()
    DS_REPLACE_TOP = auto()

    RS_PUSH = auto()
    RS_POP = auto()

    MEM_READ = auto()
    MEM_WRITE = auto()

    IO_READ = auto()
    IO_WRITE = auto()

    ALU_OP = auto()

    HALT = auto()


class Sel(Enum):
    """Selector values multiplexed onto a signal's data path."""

    # PC sources
    PC_PLUS_ONE = auto()
    PC_FROM_OPERAND = auto()
    PC_FROM_RS_TOP = auto()
    PC_FROM_OPERAND_IF_Z = auto()
    PC_FROM_OPERAND_IF_NZ = auto()
    PC_FROM_OPERAND_IF_N = auto()

    # AR sources
    AR_FROM_OPERAND = auto()
    AR_FROM_TOS = auto()
    AR_FROM_PC = auto()

    # DR sources
    DR_FROM_MEM = auto()
    DR_FROM_TOS = auto()

    # TOS sources
    TOS_FROM_ALU = auto()
    TOS_FROM_DR = auto()
    TOS_FROM_MEM = auto()
    TOS_FROM_OPERAND = auto()
    TOS_FROM_DS_TOP = auto()
    TOS_FROM_DS_POP = auto()
    TOS_FROM_IO = auto()

    # DS push source
    DS_PUSH_TOS = auto()

    # RS push source
    RS_PUSH_PC = auto()
    RS_PUSH_TOS = auto()

    # DS replace-top source
    DS_REPLACE_TOP_TOS = auto()

    # Memory write source
    MEM_WRITE_TOS = auto()
    MEM_WRITE_DR = auto()

    # IO data direction is implicit per signal; port comes from IR.operand.
    IO_PORT_FROM_OPERAND = auto()

    # ALU operations
    ALU_ADD = auto()
    ALU_SUB = auto()
    ALU_MUL = auto()
    ALU_DIV = auto()
    ALU_MOD = auto()
    ALU_NEG = auto()
    ALU_INC = auto()
    ALU_DEC = auto()

    # Micro-PC next state
    M_PC_PLUS_ONE = auto()
    M_PC_FETCH = auto()
    M_PC_DISPATCH = auto()


SignalSpec = tuple[Signal, Sel | None]


@dataclass(frozen=True)
class MicroInstr:
    """One microinstruction: signals firing in parallel within a single tick."""

    signals: tuple[SignalSpec, ...]
    label: str = ""


# ---------------------------------------------------------------------------
# Microprogram ROM addresses (named for readability and dispatch wiring).
# ---------------------------------------------------------------------------
M_FETCH = 0  # IR <- MEM[PC]; PC++
M_DECODE = 1  # m_PC <- DISPATCH(IR.opcode)

M_NOP = 2
M_HALT = 3
M_JMP = 4
M_JZ = 5
M_JNZ = 6
M_JS = 7
M_CALL_1 = 8
M_CALL_2 = 9
M_RET = 10

M_ADD = 11
M_SUB = 12
M_MUL = 13
M_DIV = 14
M_MOD = 15
M_CMP_1 = 16
M_CMP_2 = 17
M_NEG = 18
M_INC = 19
M_DEC = 20

M_LOAD_1 = 21
M_LOAD_2 = 22
M_STORE_1 = 23
M_STORE_2 = 24
M_LOADI_1 = 25
M_LOADI_2 = 26
M_STOREI_1 = 27
M_STOREI_2 = 28

M_INPUT = 29
M_OUTPUT = 30

M_PUSH = 31
M_DUP = 32
M_DROP = 33
M_SWAP = 34
M_OVER = 35


def _mi(label: str, *signals: SignalSpec) -> MicroInstr:
    return MicroInstr(signals=tuple(signals), label=label)


# A jump back to fetch is the most common terminator.
_RET_FETCH: SignalSpec = (Signal.LATCH_M_PC, Sel.M_PC_FETCH)
_NEXT: SignalSpec = (Signal.LATCH_M_PC, Sel.M_PC_PLUS_ONE)


MPROGRAM: tuple[MicroInstr, ...] = (
    # --- fetch / decode ---
    _mi(
        "FETCH",
        (Signal.LATCH_IR, None),
        (Signal.MEM_READ, None),
        (Signal.LATCH_PC, Sel.PC_PLUS_ONE),
        _NEXT,
    ),
    _mi(
        "DECODE",
        (Signal.LATCH_M_PC, Sel.M_PC_DISPATCH),
    ),
    # --- control flow ---
    _mi("NOP", _RET_FETCH),
    _mi("HALT", (Signal.HALT, None)),
    _mi(
        "JMP",
        (Signal.LATCH_PC, Sel.PC_FROM_OPERAND),
        _RET_FETCH,
    ),
    _mi(
        "JZ",
        (Signal.LATCH_PC, Sel.PC_FROM_OPERAND_IF_Z),
        _RET_FETCH,
    ),
    _mi(
        "JNZ",
        (Signal.LATCH_PC, Sel.PC_FROM_OPERAND_IF_NZ),
        _RET_FETCH,
    ),
    _mi(
        "JS",
        (Signal.LATCH_PC, Sel.PC_FROM_OPERAND_IF_N),
        _RET_FETCH,
    ),
    _mi(
        "CALL.1",
        (Signal.RS_PUSH, Sel.RS_PUSH_PC),
        _NEXT,
    ),
    _mi(
        "CALL.2",
        (Signal.LATCH_PC, Sel.PC_FROM_OPERAND),
        _RET_FETCH,
    ),
    _mi(
        "RET",
        (Signal.LATCH_PC, Sel.PC_FROM_RS_TOP),
        (Signal.RS_POP, None),
        _RET_FETCH,
    ),
    # --- ALU (binary): pop NOS, result -> TOS, set flags ---
    _mi(
        "ADD",
        (Signal.ALU_OP, Sel.ALU_ADD),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.DS_POP, None),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    _mi(
        "SUB",
        (Signal.ALU_OP, Sel.ALU_SUB),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.DS_POP, None),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    _mi(
        "MUL",
        (Signal.ALU_OP, Sel.ALU_MUL),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.DS_POP, None),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    _mi(
        "DIV",
        (Signal.ALU_OP, Sel.ALU_DIV),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.DS_POP, None),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    _mi(
        "MOD",
        (Signal.ALU_OP, Sel.ALU_MOD),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.DS_POP, None),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    # --- CMP: ( a b -- ); flags = a - b; pop both ---
    _mi(
        "CMP.1",
        (Signal.ALU_OP, Sel.ALU_SUB),
        (Signal.LATCH_FLAGS, None),
        (Signal.DS_POP, None),
        _NEXT,
    ),
    _mi(
        "CMP.2",
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_POP),
        _RET_FETCH,
    ),
    # --- ALU (unary) ---
    _mi(
        "NEG",
        (Signal.ALU_OP, Sel.ALU_NEG),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    _mi(
        "INC",
        (Signal.ALU_OP, Sel.ALU_INC),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    _mi(
        "DEC",
        (Signal.ALU_OP, Sel.ALU_DEC),
        (Signal.LATCH_TOS, Sel.TOS_FROM_ALU),
        (Signal.LATCH_FLAGS, None),
        _RET_FETCH,
    ),
    # --- memory: direct ---
    _mi(
        "LOAD.1",
        (Signal.LATCH_AR, Sel.AR_FROM_OPERAND),
        _NEXT,
    ),
    _mi(
        "LOAD.2",
        (Signal.DS_PUSH, Sel.DS_PUSH_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_MEM),
        _RET_FETCH,
    ),
    _mi(
        "STORE.1",
        (Signal.LATCH_AR, Sel.AR_FROM_OPERAND),
        _NEXT,
    ),
    _mi(
        "STORE.2",
        (Signal.MEM_WRITE, Sel.MEM_WRITE_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_POP),
        _RET_FETCH,
    ),
    # --- memory: indirect (address taken from TOS / NOS) ---
    _mi(
        "LOADI.1",
        (Signal.LATCH_AR, Sel.AR_FROM_TOS),
        _NEXT,
    ),
    _mi(
        "LOADI.2",
        (Signal.LATCH_TOS, Sel.TOS_FROM_MEM),
        _RET_FETCH,
    ),
    _mi(
        "STOREI.1",
        (Signal.LATCH_AR, Sel.AR_FROM_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_POP),
        _NEXT,
    ),
    _mi(
        "STOREI.2",
        (Signal.MEM_WRITE, Sel.MEM_WRITE_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_POP),
        _RET_FETCH,
    ),
    # --- I/O ---
    _mi(
        "INPUT",
        (Signal.DS_PUSH, Sel.DS_PUSH_TOS),
        (Signal.IO_READ, Sel.IO_PORT_FROM_OPERAND),
        (Signal.LATCH_TOS, Sel.TOS_FROM_IO),
        _RET_FETCH,
    ),
    _mi(
        "OUTPUT",
        (Signal.IO_WRITE, Sel.IO_PORT_FROM_OPERAND),
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_POP),
        _RET_FETCH,
    ),
    # --- stack manipulation ---
    _mi(
        "PUSH",
        (Signal.DS_PUSH, Sel.DS_PUSH_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_OPERAND),
        _RET_FETCH,
    ),
    _mi(
        "DUP",
        (Signal.DS_PUSH, Sel.DS_PUSH_TOS),
        _RET_FETCH,
    ),
    _mi(
        "DROP",
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_POP),
        _RET_FETCH,
    ),
    _mi(
        "SWAP",
        (Signal.DS_REPLACE_TOP, Sel.DS_REPLACE_TOP_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_TOP),
        _RET_FETCH,
    ),
    _mi(
        "OVER",
        (Signal.DS_PUSH, Sel.DS_PUSH_TOS),
        (Signal.LATCH_TOS, Sel.TOS_FROM_DS_TOP),
        _RET_FETCH,
    ),
)


DISPATCH: dict[Opcode, int] = {
    Opcode.NOP: M_NOP,
    Opcode.HALT: M_HALT,
    Opcode.JMP: M_JMP,
    Opcode.JZ: M_JZ,
    Opcode.JNZ: M_JNZ,
    Opcode.JS: M_JS,
    Opcode.CALL: M_CALL_1,
    Opcode.RET: M_RET,
    Opcode.ADD: M_ADD,
    Opcode.SUB: M_SUB,
    Opcode.MUL: M_MUL,
    Opcode.DIV: M_DIV,
    Opcode.MOD: M_MOD,
    Opcode.CMP: M_CMP_1,
    Opcode.NEG: M_NEG,
    Opcode.INC: M_INC,
    Opcode.DEC: M_DEC,
    Opcode.LOAD: M_LOAD_1,
    Opcode.STORE: M_STORE_1,
    Opcode.LOADI: M_LOADI_1,
    Opcode.STOREI: M_STOREI_1,
    Opcode.INPUT: M_INPUT,
    Opcode.OUTPUT: M_OUTPUT,
    Opcode.PUSH: M_PUSH,
    Opcode.DUP: M_DUP,
    Opcode.DROP: M_DROP,
    Opcode.SWAP: M_SWAP,
    Opcode.OVER: M_OVER,
}
