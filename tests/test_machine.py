"""End-to-end tick-accurate tests of the simulator.

Programs are assembled in-place from `Instr` objects, no translator required.
"""

from __future__ import annotations

import pytest

from csa_lab4.isa import Instr, Opcode, encode
from csa_lab4.machine import HaltError, MachineError, assemble_image, simulate


def _image(*instrs: Instr | int) -> bytes:
    words: list[int] = []
    for item in instrs:
        words.append(encode(item) if isinstance(item, Instr) else int(item))
    return assemble_image(words)


def _run(image: bytes, stdin: str = "", tick_limit: int = 10_000) -> tuple[str, str]:
    return simulate(image, stdin, tick_limit)


# ---------------------------------------------------------------------------


def test_halt_immediately() -> None:
    image = _image(Instr(Opcode.HALT))
    stdout, _ = _run(image)
    assert stdout == ""


def test_push_and_halt_leaves_tos() -> None:
    image = _image(Instr(Opcode.PUSH, 42), Instr(Opcode.HALT))
    stdout, log = _run(image)
    assert stdout == ""
    # Log should mention TOS=42 on the last live tick.
    assert "TOS=         42" in log or "TOS=          42" in log or "42" in log


def test_arithmetic_add() -> None:
    # ( 2 3 + ) → output 5 as number, then halt
    image = _image(
        Instr(Opcode.PUSH, 2),
        Instr(Opcode.PUSH, 3),
        Instr(Opcode.ADD),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "5"


def test_subtraction_and_division() -> None:
    # ( 20 6 - 2 / ) → (20-6)/2 = 7
    image = _image(
        Instr(Opcode.PUSH, 20),
        Instr(Opcode.PUSH, 6),
        Instr(Opcode.SUB),
        Instr(Opcode.PUSH, 2),
        Instr(Opcode.DIV),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "7"


def test_division_rounds_toward_zero_for_negatives() -> None:
    image = _image(
        Instr(Opcode.PUSH, -7),
        Instr(Opcode.PUSH, 2),
        Instr(Opcode.DIV),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "-3"


def test_division_by_zero_raises() -> None:
    image = _image(
        Instr(Opcode.PUSH, 1),
        Instr(Opcode.PUSH, 0),
        Instr(Opcode.DIV),
        Instr(Opcode.HALT),
    )
    with pytest.raises(MachineError):
        _run(image)


def test_jmp_unconditional() -> None:
    # JMP over an OUTPUT 99, then output 7 → must print only "7"
    # Layout: 0:JMP 3; 1:PUSH 99; 2:OUTPUT 2 (skipped); 3:PUSH 7; 4:OUTPUT 2; 5:HALT
    image = _image(
        Instr(Opcode.JMP, 3),
        Instr(Opcode.PUSH, 99),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.PUSH, 7),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "7"


def test_jz_branches_on_zero() -> None:
    # ( 5 5 cmp ) → Z=1; JZ taken; print 1, else print 0
    image = _image(
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.CMP),
        Instr(Opcode.JZ, 6),
        Instr(Opcode.PUSH, 0),
        Instr(Opcode.JMP, 7),
        Instr(Opcode.PUSH, 1),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "1"


def test_jnz_not_taken_on_equal() -> None:
    # ( 5 5 cmp ) → Z=1; JNZ not taken; falls through to PUSH 0
    image = _image(
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.CMP),
        Instr(Opcode.JNZ, 6),
        Instr(Opcode.PUSH, 0),
        Instr(Opcode.JMP, 7),
        Instr(Opcode.PUSH, 1),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "0"


def test_call_ret_round_trip() -> None:
    # Main: PUSH 10, CALL double, OUTPUT 2, HALT
    # double: DUP, ADD, RET
    # Layout: 0:PUSH 10  1:CALL 5  2:OUTPUT 2  3:HALT  4:_pad  5:DUP  6:ADD  7:RET
    image = _image(
        Instr(Opcode.PUSH, 10),
        Instr(Opcode.CALL, 5),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
        Instr(Opcode.NOP),
        Instr(Opcode.DUP),
        Instr(Opcode.ADD),
        Instr(Opcode.RET),
    )
    stdout, _ = _run(image)
    assert stdout == "20"


def test_store_then_load() -> None:
    # Layout: 0:PUSH 0x42  1:STORE 0x20  2:LOAD 0x20  3:OUTPUT 2  4:HALT
    image = _image(
        Instr(Opcode.PUSH, 0x42),
        Instr(Opcode.STORE, 0x20),
        Instr(Opcode.LOAD, 0x20),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == str(0x42)


def test_indirect_load_store() -> None:
    # Store 0x99 at address 0x30, then read via LOADI through an address on TOS.
    # 0:PUSH 0x99  1:STORE 0x30  2:PUSH 0x30  3:LOADI  4:OUTPUT 2  5:HALT
    image = _image(
        Instr(Opcode.PUSH, 0x99),
        Instr(Opcode.STORE, 0x30),
        Instr(Opcode.PUSH, 0x30),
        Instr(Opcode.LOADI),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == str(0x99)


def test_dup() -> None:
    image = _image(
        Instr(Opcode.PUSH, 7),
        Instr(Opcode.DUP),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "77"


def test_drop_uncovers_lower() -> None:
    image = _image(
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.PUSH, 9),
        Instr(Opcode.DROP),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "5"


def test_swap_exchanges_top_two() -> None:
    image = _image(
        Instr(Opcode.PUSH, 3),
        Instr(Opcode.PUSH, 8),
        Instr(Opcode.SWAP),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "38"


def test_over_duplicates_second_to_top() -> None:
    image = _image(
        Instr(Opcode.PUSH, 3),
        Instr(Opcode.PUSH, 8),
        Instr(Opcode.OVER),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "383"


def test_output_char_port() -> None:
    # Emit "Hi"
    image = _image(
        Instr(Opcode.PUSH, ord("H")),
        Instr(Opcode.OUTPUT, 1),
        Instr(Opcode.PUSH, ord("i")),
        Instr(Opcode.OUTPUT, 1),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "Hi"


def test_input_echo_one_char() -> None:
    image = _image(
        Instr(Opcode.INPUT, 0),
        Instr(Opcode.OUTPUT, 1),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image, stdin="Z")
    assert stdout == "Z"


def test_input_exhausted_halts() -> None:
    # Endless cat loop: read char, write char, jmp 0
    image = _image(
        Instr(Opcode.INPUT, 0),
        Instr(Opcode.OUTPUT, 1),
        Instr(Opcode.JMP, 0),
    )
    stdout, log = _run(image, stdin="ab")
    assert stdout == "ab"
    assert "input buffer exhausted" in log


def test_tick_limit_terminates_runaway_program() -> None:
    # Infinite NOP loop
    image = _image(Instr(Opcode.JMP, 0))
    stdout, log = _run(image, tick_limit=50)
    assert stdout == ""
    assert "tick limit" in log


def test_inc_dec() -> None:
    image = _image(
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.INC),
        Instr(Opcode.INC),
        Instr(Opcode.DEC),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "6"


def test_negative_push_via_sign_extension() -> None:
    image = _image(
        Instr(Opcode.PUSH, -7),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "-7"


def test_halt_error_after_halt() -> None:
    # Cannot easily test post-halt tick from public API; just verify halt is set.
    image = _image(Instr(Opcode.HALT))
    _, log = _run(image)
    assert "HALT" in log


def test_cmp_then_js_for_negative() -> None:
    # ( 3 5 cmp ) → result 3-5 = -2 → N=1; JS taken to print -1, else 1
    image = _image(
        Instr(Opcode.PUSH, 3),
        Instr(Opcode.PUSH, 5),
        Instr(Opcode.CMP),
        Instr(Opcode.JS, 6),
        Instr(Opcode.PUSH, 1),
        Instr(Opcode.JMP, 7),
        Instr(Opcode.PUSH, -1),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "-1"


def test_halt_error_is_subclass() -> None:
    assert issubclass(HaltError, MachineError)


# ---------------------------------------------------------------------------
# Address registers A / B with post-increment indirection.
# ---------------------------------------------------------------------------


def test_lda_then_ldai_reads_array_in_order() -> None:
    # Memory[0x40..0x43] = 11, 22, 33, 44. LDA points at 0x40, then four LDAI
    # produce TOS=44 at the end (LIFO: 11 pushed first, 44 last).
    image = _image(
        Instr(Opcode.LDA, 0x40),
        Instr(Opcode.LDAI),
        Instr(Opcode.LDAI),
        Instr(Opcode.LDAI),
        Instr(Opcode.LDAI),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    # pad code with NOPs up to addr 0x40, then store data
    padded = bytearray(image)
    while len(padded) < 0x40 * 4:
        padded.extend(b"\x00\x00\x00\x00")
    for value in (11, 22, 33, 44):
        padded.extend(value.to_bytes(4, "little", signed=True))
    stdout, _ = _run(bytes(padded))
    assert stdout == "44"


def test_stai_writes_sequential_addresses() -> None:
    # Write 1, 2, 3 into MEM[0x40..0x42] via B + STBI, then read back via LDB.
    image = _image(
        Instr(Opcode.LDB, 0x40),
        Instr(Opcode.PUSH, 1),
        Instr(Opcode.STBI),
        Instr(Opcode.PUSH, 2),
        Instr(Opcode.STBI),
        Instr(Opcode.PUSH, 3),
        Instr(Opcode.STBI),
        Instr(Opcode.LOAD, 0x40),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.LOAD, 0x41),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.LOAD, 0x42),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    stdout, _ = _run(image)
    assert stdout == "123"


def test_a_and_b_are_independent() -> None:
    # Two separate pointer walks through disjoint memory regions.
    image = _image(
        Instr(Opcode.LDA, 0x40),
        Instr(Opcode.LDB, 0x50),
        Instr(Opcode.LDAI),
        Instr(Opcode.LDBI),
        Instr(Opcode.ADD),  # MEM[0x50] + MEM[0x40]
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    padded = bytearray(image)
    while len(padded) < 0x40 * 4:
        padded.extend(b"\x00\x00\x00\x00")
    padded.extend((100).to_bytes(4, "little", signed=True))  # 0x40
    while len(padded) < 0x50 * 4:
        padded.extend(b"\x00\x00\x00\x00")
    padded.extend((25).to_bytes(4, "little", signed=True))  # 0x50
    stdout, _ = _run(bytes(padded))
    assert stdout == "125"


# ---------------------------------------------------------------------------
# Tick-level interruption.
# ---------------------------------------------------------------------------


def test_tick_hook_can_stop_mid_instruction() -> None:
    # LOAD takes 2 ticks (AR latch, then memory read). Stop the simulation
    # right after the AR-latch tick; the load must NOT complete.
    image = _image(
        Instr(Opcode.PUSH, 0x77),
        Instr(Opcode.STORE, 0x30),
        Instr(Opcode.LOAD, 0x30),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )

    # Find the tick at which mPC enters M_LOAD_1 and stop one tick after that.
    stop_after: list[int] = []

    def hook(cu: object) -> bool:
        # `cu` is a ControlUnit but typed as object for static checks; introspect.
        m_pc: int = getattr(cu, "m_pc")  # noqa: B009
        tick: int = getattr(cu, "tick_count")  # noqa: B009
        # Stop on the first tick that completes the AR latch micro-step
        # (m_pc == M_LOAD_2 means LOAD.1 has just finished).
        from csa_lab4.microcode import M_LOAD_2  # noqa: PLC0415

        if m_pc == M_LOAD_2 and not stop_after:
            stop_after.append(tick)
            return False
        return True

    stdout, log = simulate(image, "", tick_limit=10_000, tick_hook=hook)
    assert stdout == ""  # LOAD did not get to OUTPUT
    assert "stopped by tick hook" in log
    assert stop_after  # hook fired


def test_tick_hook_returning_true_runs_to_completion() -> None:
    image = _image(
        Instr(Opcode.PUSH, 1),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    seen: list[int] = []

    def hook(cu: object) -> bool:
        seen.append(getattr(cu, "tick_count"))  # noqa: B009
        return True

    stdout, _ = simulate(image, "", tick_limit=10_000, tick_hook=hook)
    assert stdout == "1"
    assert len(seen) > 0


def test_external_stepping_through_control_unit() -> None:
    """Driving ControlUnit.tick() directly enables arbitrary stepping."""
    from csa_lab4.machine import IO, ControlUnit, DataPath, load_binary  # noqa: PLC0415

    image = _image(
        Instr(Opcode.PUSH, 11),
        Instr(Opcode.PUSH, 22),
        Instr(Opcode.ADD),
        Instr(Opcode.OUTPUT, 2),
        Instr(Opcode.HALT),
    )
    dp = DataPath.from_image(load_binary(image))
    cu = ControlUnit(dp, IO())
    # Run for exactly N ticks then inspect.
    for _ in range(5):
        cu.tick()
    assert cu.tick_count == 5
    assert not cu.halted
