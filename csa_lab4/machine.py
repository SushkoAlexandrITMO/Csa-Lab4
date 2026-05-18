"""Tick-accurate microcoded stack-machine simulator.

The simulator interprets the microprogram defined in :mod:`csa_lab4.microcode`.
One Python call to :meth:`ControlUnit.tick` simulates exactly one hardware
tick. Each tick samples every state register at the start of the tick
(``Snapshot``), evaluates combinatorial paths (ALU, memory read, IO peek)
from the snapshot, then commits new values to every latch at tick end.
This mirrors a synchronous design with edge-triggered registers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from csa_lab4.isa import OPERAND_BITS, OPERAND_MASK, WORD_BITS, WORD_MASK, Opcode, decode, mnemonic
from csa_lab4.microcode import DISPATCH, M_FETCH, MPROGRAM, Sel, Signal

MEMORY_SIZE: int = 1 << 16
ADDR_MASK: int = MEMORY_SIZE - 1
WORD_SIGN_BIT: int = 1 << (WORD_BITS - 1)
OPERAND_SIGN_BIT: int = 1 << (OPERAND_BITS - 1)


class MachineError(RuntimeError):
    """Base class for simulator runtime errors."""


class HaltError(MachineError):
    """Raised when the machine halts (HALT signal or input exhausted)."""


class StackError(MachineError):
    """Underflow / overflow on the data or return stack."""


def _signed_word(value: int) -> int:
    value &= WORD_MASK
    if value & WORD_SIGN_BIT:
        return value - (1 << WORD_BITS)
    return value


def _signed_operand(value: int) -> int:
    value &= OPERAND_MASK
    if value & OPERAND_SIGN_BIT:
        return value - (1 << OPERAND_BITS)
    return value


# ---------------------------------------------------------------------------
# IO subsystem (stream + port)
# ---------------------------------------------------------------------------

INPUT_PORT_CHAR: int = 0
OUTPUT_PORT_CHAR: int = 1
OUTPUT_PORT_NUM: int = 2


@dataclass
class IO:
    """Port-mapped stream IO.

    Port 0: input — one character per read, ``HaltError`` when buffer empty.
    Port 1: output — TOS interpreted as 8-bit char and appended to stdout.
    Port 2: output — TOS interpreted as signed integer, decimal text emitted.
    """

    input_buffer: list[int] = field(default_factory=list)
    output_chunks: list[str] = field(default_factory=list)

    def peek(self, port: int) -> int:
        if port != INPUT_PORT_CHAR:
            raise MachineError(f"unsupported input port: {port}")
        if not self.input_buffer:
            raise HaltError("input buffer exhausted")
        return self.input_buffer[0]

    def consume(self, port: int) -> None:
        if port != INPUT_PORT_CHAR:
            raise MachineError(f"unsupported input port: {port}")
        self.input_buffer.pop(0)

    def write(self, port: int, value: int) -> None:
        if port == OUTPUT_PORT_CHAR:
            self.output_chunks.append(chr(value & 0xFF))
        elif port == OUTPUT_PORT_NUM:
            self.output_chunks.append(str(_signed_word(value)))
        else:
            raise MachineError(f"unsupported output port: {port}")

    @property
    def output_text(self) -> str:
        return "".join(self.output_chunks)


# ---------------------------------------------------------------------------
# DataPath
# ---------------------------------------------------------------------------


@dataclass
class DataPath:
    memory: list[int]
    ds: list[int] = field(default_factory=list)
    rs: list[int] = field(default_factory=list)
    tos: int = 0
    pc: int = 0
    ar: int = 0
    dr: int = 0
    ir: int = 0
    flag_z: bool = False
    flag_n: bool = False

    @classmethod
    def from_image(cls, words: list[int]) -> DataPath:
        memory = [0] * MEMORY_SIZE
        if len(words) > MEMORY_SIZE:
            raise MachineError(f"image too large: {len(words)} > {MEMORY_SIZE}")
        for i, w in enumerate(words):
            memory[i] = _signed_word(w)
        return cls(memory=memory)


@dataclass(frozen=True)
class Snapshot:
    """Sampled DataPath state at the start of a tick."""

    pc: int
    ar: int
    dr: int
    ir: int
    tos: int
    ds_top: int
    rs_top: int
    flag_z: bool
    flag_n: bool
    operand_unsigned: int
    operand_signed: int
    opcode_raw: int
    mem_at_pc: int
    mem_at_ar: int


def _snapshot(dp: DataPath) -> Snapshot:
    ir = dp.ir & WORD_MASK
    operand_raw = ir & OPERAND_MASK
    return Snapshot(
        pc=dp.pc & ADDR_MASK,
        ar=dp.ar & ADDR_MASK,
        dr=_signed_word(dp.dr),
        ir=ir,
        tos=_signed_word(dp.tos),
        ds_top=dp.ds[-1] if dp.ds else 0,
        rs_top=dp.rs[-1] if dp.rs else 0,
        flag_z=dp.flag_z,
        flag_n=dp.flag_n,
        operand_unsigned=operand_raw,
        operand_signed=_signed_operand(operand_raw),
        opcode_raw=(ir >> 24) & 0xFF,
        mem_at_pc=dp.memory[dp.pc & ADDR_MASK],
        mem_at_ar=dp.memory[dp.ar & ADDR_MASK],
    )


# ---------------------------------------------------------------------------
# Control Unit
# ---------------------------------------------------------------------------


class ControlUnit:
    def __init__(self, data_path: DataPath, io: IO) -> None:
        self.data_path = data_path
        self.io = io
        self.m_pc: int = M_FETCH
        self.tick_count: int = 0
        self.halted: bool = False
        self.log_lines: list[str] = []

    # ----- ALU --------------------------------------------------------------

    @staticmethod
    def _alu(sel: Sel | None, snap: Snapshot) -> int:
        a = snap.tos
        b = snap.ds_top
        if sel is Sel.ALU_ADD:
            return _signed_word(b + a)
        if sel is Sel.ALU_SUB:
            return _signed_word(b - a)
        if sel is Sel.ALU_MUL:
            return _signed_word(b * a)
        if sel is Sel.ALU_DIV:
            if a == 0:
                raise MachineError("division by zero")
            # truncated toward zero (Python // truncates toward -inf, adjust)
            q = abs(b) // abs(a)
            if (a < 0) ^ (b < 0):
                q = -q
            return _signed_word(q)
        if sel is Sel.ALU_MOD:
            if a == 0:
                raise MachineError("modulo by zero")
            q = abs(b) // abs(a)
            if (a < 0) ^ (b < 0):
                q = -q
            return _signed_word(b - q * a)
        if sel is Sel.ALU_NEG:
            return _signed_word(-a)
        if sel is Sel.ALU_INC:
            return _signed_word(a + 1)
        if sel is Sel.ALU_DEC:
            return _signed_word(a - 1)
        raise MachineError(f"bad ALU op selector: {sel}")

    # ----- source selectors -------------------------------------------------

    @staticmethod
    def _next_pc(sel: Sel | None, snap: Snapshot) -> int:
        if sel is Sel.PC_PLUS_ONE:
            return (snap.pc + 1) & ADDR_MASK
        if sel is Sel.PC_FROM_OPERAND:
            return snap.operand_unsigned & ADDR_MASK
        if sel is Sel.PC_FROM_RS_TOP:
            return snap.rs_top & ADDR_MASK
        if sel is Sel.PC_FROM_OPERAND_IF_Z:
            return snap.operand_unsigned & ADDR_MASK if snap.flag_z else snap.pc
        if sel is Sel.PC_FROM_OPERAND_IF_NZ:
            return snap.operand_unsigned & ADDR_MASK if not snap.flag_z else snap.pc
        if sel is Sel.PC_FROM_OPERAND_IF_N:
            return snap.operand_unsigned & ADDR_MASK if snap.flag_n else snap.pc
        raise MachineError(f"bad PC selector: {sel}")

    @staticmethod
    def _next_ar(sel: Sel | None, snap: Snapshot) -> int:
        if sel is Sel.AR_FROM_OPERAND:
            return snap.operand_unsigned & ADDR_MASK
        if sel is Sel.AR_FROM_TOS:
            return snap.tos & ADDR_MASK
        if sel is Sel.AR_FROM_PC:
            return snap.pc & ADDR_MASK
        raise MachineError(f"bad AR selector: {sel}")

    @staticmethod
    def _next_dr(sel: Sel | None, snap: Snapshot) -> int:
        if sel is Sel.DR_FROM_MEM:
            return snap.mem_at_ar
        if sel is Sel.DR_FROM_TOS:
            return snap.tos
        raise MachineError(f"bad DR selector: {sel}")

    @staticmethod
    def _next_m_pc(sel: Sel | None, snap: Snapshot, current_m_pc: int) -> int:
        if sel is Sel.M_PC_PLUS_ONE:
            return current_m_pc + 1
        if sel is Sel.M_PC_FETCH:
            return M_FETCH
        if sel is Sel.M_PC_DISPATCH:
            opcode = Opcode(snap.opcode_raw)
            return DISPATCH[opcode]
        raise MachineError(f"bad m_PC selector: {sel}")

    @staticmethod
    def _tos_source(
        sel: Sel | None,
        snap: Snapshot,
        alu_result: int | None,
        io_data: int | None,
    ) -> tuple[int, bool]:
        """Return (new TOS, did-pop-ds flag)."""
        if sel is Sel.TOS_FROM_ALU:
            assert alu_result is not None, "LATCH_TOS=ALU without ALU_OP in same tick"
            return alu_result, False
        if sel is Sel.TOS_FROM_DR:
            return snap.dr, False
        if sel is Sel.TOS_FROM_MEM:
            return snap.mem_at_ar, False
        if sel is Sel.TOS_FROM_OPERAND:
            return snap.operand_signed, False
        if sel is Sel.TOS_FROM_DS_TOP:
            return snap.ds_top, False
        if sel is Sel.TOS_FROM_DS_POP:
            return snap.ds_top, True
        if sel is Sel.TOS_FROM_IO:
            assert io_data is not None, "LATCH_TOS=IO without IO_READ in same tick"
            return io_data, False
        raise MachineError(f"bad TOS selector: {sel}")

    # ----- main tick --------------------------------------------------------

    def tick(self) -> None:
        if self.halted:
            raise HaltError("attempt to tick a halted machine")

        micro = MPROGRAM[self.m_pc]
        snap = _snapshot(self.data_path)

        # Phase A: evaluate combinatorial outputs.
        alu_result: int | None = None
        io_data: int | None = None
        io_consume_port: int | None = None
        for sig, sel in micro.signals:
            if sig is Signal.ALU_OP:
                alu_result = self._alu(sel, snap)
            elif sig is Signal.IO_READ:
                io_data = self.io.peek(snap.operand_unsigned & 0xFF)
                io_consume_port = snap.operand_unsigned & 0xFF

        # Phase B: compute new values for every potentially-latched destination.
        next_pc = snap.pc
        next_ar = snap.ar
        next_dr = snap.dr
        next_ir = snap.ir
        next_tos = snap.tos
        next_z = snap.flag_z
        next_n = snap.flag_n
        next_m_pc = self.m_pc

        ds_push = False
        ds_pop_count = 0
        ds_replace_top = False
        rs_push_value: int | None = None
        rs_pop = False
        mem_write_value: int | None = None
        io_writes: list[tuple[int, int]] = []
        halt = False

        for sig, sel in micro.signals:
            if sig is Signal.LATCH_PC:
                next_pc = self._next_pc(sel, snap)
            elif sig is Signal.LATCH_AR:
                next_ar = self._next_ar(sel, snap)
            elif sig is Signal.LATCH_DR:
                next_dr = self._next_dr(sel, snap)
            elif sig is Signal.LATCH_IR:
                next_ir = snap.mem_at_pc & WORD_MASK
            elif sig is Signal.LATCH_TOS:
                next_tos, popped = self._tos_source(sel, snap, alu_result, io_data)
                if popped:
                    ds_pop_count += 1
            elif sig is Signal.LATCH_FLAGS:
                assert alu_result is not None, "LATCH_FLAGS without ALU_OP in same tick"
                next_z = alu_result == 0
                next_n = alu_result < 0
            elif sig is Signal.LATCH_M_PC:
                next_m_pc = self._next_m_pc(sel, snap, self.m_pc)
            elif sig is Signal.DS_PUSH:
                ds_push = True
            elif sig is Signal.DS_POP:
                ds_pop_count += 1
            elif sig is Signal.DS_REPLACE_TOP:
                ds_replace_top = True
            elif sig is Signal.RS_PUSH:
                rs_push_value = snap.pc if sel is Sel.RS_PUSH_PC else snap.tos
            elif sig is Signal.RS_POP:
                rs_pop = True
            elif sig is Signal.MEM_WRITE:
                mem_write_value = snap.tos if sel is Sel.MEM_WRITE_TOS else snap.dr
            elif sig is Signal.IO_WRITE:
                io_writes.append((snap.operand_unsigned & 0xFF, snap.tos))
            elif sig is Signal.HALT:
                halt = True
            elif sig in (Signal.MEM_READ, Signal.ALU_OP, Signal.IO_READ):
                pass  # already handled in phase A or via source selectors

        # Phase C: commit.
        self.data_path.pc = next_pc & ADDR_MASK
        self.data_path.ar = next_ar & ADDR_MASK
        self.data_path.dr = _signed_word(next_dr)
        self.data_path.ir = next_ir & WORD_MASK
        self.data_path.tos = _signed_word(next_tos)
        self.data_path.flag_z = next_z
        self.data_path.flag_n = next_n

        if ds_push:
            self.data_path.ds.append(snap.tos)
        if ds_replace_top:
            if not self.data_path.ds:
                raise StackError("DS_REPLACE_TOP on empty data stack")
            self.data_path.ds[-1] = snap.tos
        for _ in range(ds_pop_count):
            if not self.data_path.ds:
                raise StackError("DS_POP on empty data stack")
            self.data_path.ds.pop()
        if rs_push_value is not None:
            self.data_path.rs.append(rs_push_value & ADDR_MASK)
        if rs_pop:
            if not self.data_path.rs:
                raise StackError("RS_POP on empty return stack")
            self.data_path.rs.pop()
        if mem_write_value is not None:
            self.data_path.memory[snap.ar] = _signed_word(mem_write_value)
        for port, value in io_writes:
            self.io.write(port, value)
        if io_consume_port is not None:
            self.io.consume(io_consume_port)

        self.m_pc = next_m_pc
        self.tick_count += 1
        self._log_tick(micro, snap)

        if halt:
            self.halted = True
            raise HaltError("HALT")

    # ----- logging ---------------------------------------------------------

    def _log_tick(self, micro: object, pre_snap: Snapshot) -> None:
        dp = self.data_path
        annotation = ""
        # On the DECODE step we know the upcoming opcode — annotate mnemonic.
        if pre_snap.opcode_raw in (op.value for op in Opcode) and self.m_pc in DISPATCH.values():
            try:
                instr = decode(dp.ir)
                annotation = f"  ; {mnemonic(instr)}"
            except ValueError:
                annotation = ""
        label = getattr(micro, "label", "?")
        line = (
            f"T{self.tick_count:05d}  m_PC={self.m_pc:02d} {label:<10s}  "
            f"PC={dp.pc:04X}  IR={dp.ir & WORD_MASK:08X}  "
            f"TOS={dp.tos:>11d}  "
            f"DS[{len(dp.ds)}]  RS[{len(dp.rs)}]  "
            f"Z={int(dp.flag_z)} N={int(dp.flag_n)}"
            f"{annotation}"
        )
        self.log_lines.append(line)


# ---------------------------------------------------------------------------
# Binary loader and simulator entry point
# ---------------------------------------------------------------------------


def load_binary(blob: bytes) -> list[int]:
    if len(blob) % 4 != 0:
        raise MachineError(f"binary size {len(blob)} is not a multiple of 4 bytes")
    return [int.from_bytes(blob[i : i + 4], byteorder="little", signed=True) for i in range(0, len(blob), 4)]


def simulate(binary: bytes, stdin_data: str, tick_limit: int) -> tuple[str, str]:
    image = load_binary(binary)
    dp = DataPath.from_image(image)
    io = IO(input_buffer=[ord(c) for c in stdin_data])
    cu = ControlUnit(dp, io)

    halt_reason = ""
    try:
        while cu.tick_count < tick_limit:
            cu.tick()
    except HaltError as exc:
        halt_reason = str(exc)
    if not cu.halted and cu.tick_count >= tick_limit:
        halt_reason = f"tick limit reached ({tick_limit})"

    log_text = "\n".join(cu.log_lines)
    if halt_reason:
        log_text += f"\n; halted: {halt_reason}\n"
    return io.output_text, log_text


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


def assemble_image(words: list[int]) -> bytes:
    """Helper: pack a list of 32-bit words into a little-endian binary blob."""
    return b"".join(int(w & WORD_MASK).to_bytes(4, byteorder="little", signed=False) for w in words)


if __name__ == "__main__":
    sys.exit(main())
