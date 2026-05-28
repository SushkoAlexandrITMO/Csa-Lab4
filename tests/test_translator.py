"""End-to-end tests for the translator: lisp source -> binary -> output."""

from __future__ import annotations

import pytest

from csa_lab4.machine import simulate
from csa_lab4.translator import TranslationError, parse, tokenize, translate


def _run(source: str, stdin: str = "", tick_limit: int = 100_000) -> str:
    binary, _ = translate(source)
    stdout, _ = simulate(binary, stdin, tick_limit)
    return stdout


# ---------------------------------------------------------------------------
# Lexer / parser
# ---------------------------------------------------------------------------


def test_lex_basic() -> None:
    toks = tokenize("(+ 1 2)")
    kinds = [t.kind for t in toks]
    assert kinds == ["lpar", "sym", "int", "int", "rpar"]


def test_lex_string_with_escape() -> None:
    toks = tokenize('"hi\\n"')
    assert toks[0].kind == "str"
    assert toks[0].value == "hi\n"


def test_lex_comments_are_skipped() -> None:
    toks = tokenize("; a comment\n(+ 1 2)")
    assert [t.value for t in toks] == ["(", "+", "1", "2", ")"]


def test_parse_nested() -> None:
    program = parse(tokenize("(if (= x 0) 1 2)"))
    assert len(program) == 1


def test_parse_unbalanced_parens_raises() -> None:
    with pytest.raises(TranslationError):
        parse(tokenize("(+ 1 2"))


# ---------------------------------------------------------------------------
# Compilation + execution
# ---------------------------------------------------------------------------


def test_print_literal_int() -> None:
    assert _run("(print-int 42)") == "42"


def test_print_negative_int() -> None:
    assert _run("(print-int -17)") == "-17"


def test_addition() -> None:
    assert _run("(print-int (+ 2 3))") == "5"


def test_subtraction_uses_forth_order() -> None:
    # (- a b) compiles as: push a, push b, SUB.  SUB = NOS - TOS = a - b.
    assert _run("(print-int (- 10 3))") == "7"


def test_chained_arithmetic() -> None:
    # ( 1 + 2 ) * ( 5 - 3 ) = 6
    assert _run("(print-int (* (+ 1 2) (- 5 3)))") == "6"


def test_division_and_modulo() -> None:
    assert _run("(print-int (/ 23 4))") == "5"
    assert _run("(print-int (mod 23 4))") == "3"


def test_if_then_branch() -> None:
    assert _run("(print-int (if (= 1 1) 7 9))") == "7"


def test_if_else_branch() -> None:
    assert _run("(print-int (if (= 1 2) 7 9))") == "9"


def test_less_than_materialises() -> None:
    assert _run("(print-int (if (< 3 5) 100 200))") == "100"
    assert _run("(print-int (if (< 5 3) 100 200))") == "200"


def test_greater_than_materialises() -> None:
    assert _run("(print-int (if (> 5 3) 100 200))") == "100"
    assert _run("(print-int (if (> 3 5) 100 200))") == "200"
    assert _run("(print-int (if (> 3 3) 100 200))") == "200"  # strict


def test_global_setq_then_read() -> None:
    assert _run("(setq x 5) (print-int x)") == "5"


def test_global_setq_returns_value() -> None:
    # setq is an expression, so we can directly print its result.
    assert _run("(print-int (setq y 11))") == "11"


def test_global_update() -> None:
    assert _run("(setq x 1) (setq x (+ x 10)) (print-int x)") == "11"


def test_progn_yields_last() -> None:
    assert _run("(print-int (progn 1 2 3))") == "3"


def test_let_binding_simple() -> None:
    assert _run("(print-int (let ((x 10) (y 20)) (+ x y)))") == "30"


def test_let_nested() -> None:
    src = "(print-int (let ((x 3)) (let ((y 4)) (+ x y))))"
    assert _run(src) == "7"


def test_defun_square() -> None:
    src = "(defun sq (x) (* x x)) (print-int (sq 7))"
    assert _run(src) == "49"


def test_defun_two_args() -> None:
    src = "(defun add2 (a b) (+ a b)) (print-int (add2 11 22))"
    assert _run(src) == "33"


def test_defun_tail_recursive_factorial() -> None:
    src = """
    (defun fact-iter (n acc)
      (if (= n 0)
          acc
          (fact-iter (- n 1) (* acc n))))
    (defun fact (n) (fact-iter n 1))
    (print-int (fact 5))
    """
    assert _run(src) == "120"


def test_defun_returns_expression_value() -> None:
    # The whole defun call result becomes the expression value.
    src = "(defun double (x) (+ x x)) (print-int (* 10 (double 4)))"
    assert _run(src) == "80"


def test_print_char() -> None:
    src = "(print-char 72) (print-char 105)"  # "Hi"
    assert _run(src) == "Hi"


def test_read_then_echo() -> None:
    # (print-char (read-char)) one round
    src = "(print-char (read-char))"
    assert _run(src, stdin="Z") == "Z"


def test_halt_form() -> None:
    src = "(print-int 1) (halt) (print-int 999)"
    assert _run(src) == "1"


def test_undefined_variable_raises() -> None:
    with pytest.raises(TranslationError):
        _run("(print-int undefined-name)")


def test_unknown_function_raises() -> None:
    with pytest.raises(TranslationError):
        _run("(no-such-fn 1)")


def test_defun_three_params_via_pick() -> None:
    # Three params force depth-2 access (via PICK) for the leftmost arg.
    src = "(defun add3 (a b c) (+ a (+ b c))) (print-int (add3 1 2 3))"
    assert _run(src) == "6"


def test_listing_contains_reset_vector_and_data() -> None:
    _, listing = translate("(setq x 1) (print-int x)")
    assert "reset vector" in listing
    assert "data segment" in listing
    assert "code segment" in listing


def test_string_literal_placed_in_data_segment() -> None:
    # The string lives in data; the expression value is its address.
    _, listing = translate('(print-int "Hi")')
    assert "DATA  2" in listing  # length word for "Hi"
    assert "DATA  72" in listing  # 'H'
    assert "DATA  105" in listing  # 'i'


def test_load_explicit_address() -> None:
    src = "(setq x 99) (print-int (load 16))"  # 16 = DATA_SEGMENT_START
    assert _run(src) == "99"


# ---------------------------------------------------------------------------
# Bitwise primitives.
# ---------------------------------------------------------------------------


def test_bitand() -> None:
    assert _run("(print-int (bitand 12 10))") == "8"


def test_bitor() -> None:
    assert _run("(print-int (bitor 12 10))") == "14"


def test_bitxor() -> None:
    assert _run("(print-int (bitxor 12 10))") == "6"


def test_bitnot() -> None:
    assert _run("(print-int (bitnot 0))") == "-1"


def test_bitand_even_odd() -> None:
    assert _run("(print-int (if (= (bitand 7 1) 0) 100 200))") == "200"
    assert _run("(print-int (if (= (bitand 8 1) 0) 100 200))") == "100"


# ---------------------------------------------------------------------------
# A / B register access from lisp.
# ---------------------------------------------------------------------------


def test_set_a_get_a() -> None:
    assert _run("(set-a 55) (print-int (get-a))") == "55"


def test_set_b_get_b() -> None:
    assert _run("(set-b 66) (print-int (get-b))") == "66"


def test_set_a_returns_value() -> None:
    assert _run("(print-int (set-a 99))") == "99"


# ---------------------------------------------------------------------------
# Carry-aware arithmetic from lisp (used by the double_precision golden test).
# ---------------------------------------------------------------------------


def test_adc_propagates_carry_from_preceding_add() -> None:
    src = """
    (setq alo 3000000000)
    (setq blo 2000000000)
    (setq lo (+ alo blo))
    (setq hi (adc 1 2))
    (print-int hi)
    """
    assert _run(src) == "4"
