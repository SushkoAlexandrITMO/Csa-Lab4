"""Lisp -> binary translator.

Pipeline:
  text  ->  tokens  ->  AST  ->  IR (labelled instructions + data words)
        ->  binary image + .lst listing

The compiler tracks a *stack layout* — an ordered list of named slots
representing the current shape of the data stack at each point in the
emitted code. Variable references resolve to ``DUP`` (depth 0),
``OVER`` (depth 1), or ``PICK depth`` for arbitrary deeper accesses.

Top-level forms:
  (setq name expr)                define / update a global, expression-valued
  (defun name (a b) body)         define a procedure
  (if cond then else)             every form is an expression
  (progn a b c)                   sequence; value of the last expression
  (let ((x e) (y e2)) body)       lexical bindings, also expression-valued
  arithmetic   + - * / mod neg
  comparison   = < >              materialize 1 (true) or 0 (false)
  primitives   load store print-char print-int read-char halt
  literals     integers, strings  ("..."  -> pstr in data segment)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from csa_lab4.isa import OPERAND_BITS, Instr, Opcode, encode, mnemonic

# Reserve the low addresses for the entry-point jump and a small interrupt
# vector area (not used in the stream variant, but kept symmetric with the
# memory layout described in the report).
DATA_SEGMENT_START: int = 0x0010
DEFAULT_CODE_BASE: int = 0x1000

OPERAND_LIMIT_POS: int = (1 << (OPERAND_BITS - 1)) - 1
OPERAND_LIMIT_NEG: int = -(1 << (OPERAND_BITS - 1))


class TranslationError(Exception):
    """Raised when the input program cannot be compiled."""


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str  # "lpar" | "rpar" | "int" | "str" | "sym"
    value: str
    line: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    while i < len(source):
        ch = source[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        if ch == ";":  # line comment
            while i < len(source) and source[i] != "\n":
                i += 1
            continue
        if ch == "(":
            tokens.append(Token("lpar", "(", line))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token("rpar", ")", line))
            i += 1
            continue
        if ch == '"':
            j = i + 1
            buf: list[str] = []
            while j < len(source) and source[j] != '"':
                if source[j] == "\\" and j + 1 < len(source):
                    escape = source[j + 1]
                    buf.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(escape, escape))
                    j += 2
                    continue
                buf.append(source[j])
                j += 1
            if j >= len(source):
                raise TranslationError(f"unterminated string at line {line}")
            tokens.append(Token("str", "".join(buf), line))
            i = j + 1
            continue
        # Number or symbol
        j = i
        while j < len(source) and not source[j].isspace() and source[j] not in '()";':
            j += 1
        raw = source[i:j]
        if _looks_like_int(raw):
            tokens.append(Token("int", raw, line))
        else:
            tokens.append(Token("sym", raw, line))
        i = j
    return tokens


def _looks_like_int(text: str) -> bool:
    if not text:
        return False
    s = text[1:] if text[0] in "+-" else text
    return bool(s) and all(c.isdigit() for c in s)


# ---------------------------------------------------------------------------
# Parser (AST = Python lists of atoms; atoms are int | str-literal | Symbol)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class StringLit:
    text: str


Atom = int | Symbol | StringLit
Form = Atom | list["Form"]


def parse(tokens: list[Token]) -> list[Form]:
    pos = 0
    program: list[Form] = []
    while pos < len(tokens):
        form, pos = _parse_form(tokens, pos)
        program.append(form)
    return program


def _parse_form(tokens: list[Token], pos: int) -> tuple[Form, int]:
    tok = tokens[pos]
    if tok.kind == "lpar":
        pos += 1
        items: list[Form] = []
        while pos < len(tokens) and tokens[pos].kind != "rpar":
            sub, pos = _parse_form(tokens, pos)
            items.append(sub)
        if pos >= len(tokens):
            raise TranslationError(f"unbalanced '(' opened at line {tok.line}")
        return items, pos + 1
    if tok.kind == "rpar":
        raise TranslationError(f"unexpected ')' at line {tok.line}")
    if tok.kind == "int":
        return int(tok.value), pos + 1
    if tok.kind == "str":
        return StringLit(tok.value), pos + 1
    if tok.kind == "sym":
        return Symbol(tok.value), pos + 1
    raise TranslationError(f"unknown token kind {tok.kind!r} at line {tok.line}")


# ---------------------------------------------------------------------------
# IR: labelled instructions and data words
# ---------------------------------------------------------------------------


@dataclass
class CodeItem:
    """A single addressable code word.

    For instructions whose operand is a symbolic label, ``operand_label`` is
    set; ``operand`` is filled in by the linker. ``label`` may be set for
    items that are jump/call targets.
    """

    opcode: Opcode
    operand: int = 0
    operand_label: str | None = None
    label: str | None = None
    source: str = ""  # human-readable annotation for the .lst listing


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


PRIMITIVE_BINARY: dict[str, Opcode] = {
    "+": Opcode.ADD,
    "-": Opcode.SUB,
    "*": Opcode.MUL,
    "/": Opcode.DIV,
    "mod": Opcode.MOD,
}

PRIMITIVE_UNARY: dict[str, Opcode] = {
    "neg": Opcode.NEG,
    "1+": Opcode.INC,
    "1-": Opcode.DEC,
}

COMPARISONS: dict[str, Opcode] = {
    # All implemented through CMP + conditional materialisation.
    "=": Opcode.JZ,  # equal -> Z=1
    "<": Opcode.JS,  # a < b means a - b is negative -> N=1
    ">": Opcode.JNZ,  # placeholder, treated specially
}


@dataclass
class Compiler:
    code: list[CodeItem] = field(default_factory=list)
    data: list[int] = field(default_factory=list)  # words in the data segment

    # Variable name -> data-segment address.
    globals_: dict[str, int] = field(default_factory=dict)
    # Function name -> code label.
    functions: dict[str, str] = field(default_factory=dict)
    # Stack layout: each entry is a slot name ("" for anonymous values).
    stack: list[str] = field(default_factory=list)

    _label_counter: int = 0

    # ------------------------------------------------------------------ utils
    def emit(self, opcode: Opcode, operand: int = 0, *, operand_label: str | None = None, source: str = "") -> None:
        self.code.append(CodeItem(opcode=opcode, operand=operand, operand_label=operand_label, source=source))

    def attach_label(self, label: str) -> None:
        # Attach to the next emitted instruction. We materialise via a NOP if
        # the next instruction hasn't been emitted yet.
        if self.code and self.code[-1].label is None:
            # Attach to a new NOP so we always have a stable target word.
            pass
        # Simpler: always create a fresh NOP that carries the label.
        self.code.append(CodeItem(opcode=Opcode.NOP, label=label, source=f"{label}:"))

    def gen_label(self, prefix: str) -> str:
        self._label_counter += 1
        return f"{prefix}_{self._label_counter}"

    def alloc_data_word(self, initial: int = 0) -> int:
        addr = DATA_SEGMENT_START + len(self.data)
        self.data.append(initial)
        return addr

    def push_slot(self, name: str = "") -> None:
        self.stack.append(name)

    def pop_slot(self) -> None:
        if not self.stack:
            raise TranslationError("compiler stack underflow (internal)")
        self.stack.pop()

    def find_slot_depth(self, name: str) -> int | None:
        for depth, slot in enumerate(reversed(self.stack)):
            if slot == name:
                return depth
        return None

    # --------------------------------------------------------------- compile
    def compile_program(self, program: list[Form]) -> None:
        # First pass: collect global variable definitions and function
        # declarations so forward references resolve.
        for form in program:
            self._collect_top_level(form)

        # Second pass: compile.  Memory layout: the linker plants a synthesised
        # ``JMP _start`` at address 0 (the reset vector), so the first emitted
        # code item is the start of user code at ``code_base``.
        self.attach_label("_start")
        for form in program:
            if _is_call(form, "defun"):
                self.compile_defun(form)  # type: ignore[arg-type]
                continue
            self.compile_expr(form)
            # Drop the expression's value: top-level expressions are evaluated
            # for their side effects.
            self.emit(Opcode.DROP, source="DROP  ; discard top-level value")
            self.pop_slot()
        self.emit(Opcode.HALT, source="HALT")

    def _collect_top_level(self, form: Form) -> None:
        if _is_call(form, "setq"):
            assert isinstance(form, list)
            if len(form) != 3:
                raise TranslationError("(setq name value) takes 2 args")
            name = _expect_symbol(form[1])
            if name not in self.globals_:
                self.globals_[name] = self.alloc_data_word()
        elif _is_call(form, "defun"):
            assert isinstance(form, list)
            if len(form) < 4:
                raise TranslationError("(defun name (params) body...)")
            name = _expect_symbol(form[1])
            self.functions[name] = f"fn_{name}"

    # ---- top-level forms ----
    def compile_defun(self, form: list[Form]) -> None:
        name = _expect_symbol(form[1])
        params = form[2]
        if not isinstance(params, list):
            raise TranslationError(f"defun {name}: param list must be a list")
        param_names = [_expect_symbol(p) for p in params]
        body = form[3:]

        # Skip past the function body when running top-to-bottom.
        end_label = self.gen_label("defun_end")
        self.emit(Opcode.JMP, operand_label=end_label, source=f"JMP {end_label}  ; skip body of {name}")

        # Function entry. Args are on top of the data stack: layout = params.
        self.attach_label(self.functions[name])
        saved_stack = self.stack[:]
        self.stack = list(param_names)

        # Body — sequence of expressions, value of the last one is the return.
        for i, expr in enumerate(body):
            self.compile_expr(expr)
            if i < len(body) - 1:
                self.emit(Opcode.DROP, source="DROP  ; sequence non-final value")
                self.pop_slot()

        # Epilogue: result is on top. Args are buried under it. NIP each.
        for _ in param_names:
            self.emit(Opcode.SWAP, source="SWAP  ; bury result")
            self.emit(Opcode.DROP, source="DROP  ; pop one arg")
        self.emit(Opcode.RET, source="RET")
        self.stack = saved_stack
        self.attach_label(end_label)

    # ---- expressions ----
    def compile_expr(self, form: Form) -> None:
        if isinstance(form, int):
            self.compile_int_literal(form)
            return
        if isinstance(form, StringLit):
            self.compile_string_literal(form.text)
            return
        if isinstance(form, Symbol):
            self.compile_variable(form.name)
            return
        if isinstance(form, list):
            self.compile_list(form)
            return
        raise TranslationError(f"cannot compile {form!r}")

    def compile_int_literal(self, value: int) -> None:
        if OPERAND_LIMIT_NEG <= value <= OPERAND_LIMIT_POS:
            self.emit(Opcode.PUSH, value, source=f"PUSH {value}")
            self.push_slot()
            return
        # Too wide for immediate — spill into the data segment.
        addr = self.alloc_data_word(value & ((1 << 32) - 1))
        self.emit(Opcode.LOAD, addr, source=f"LOAD 0x{addr:04X}  ; literal {value}")
        self.push_slot()

    def compile_string_literal(self, text: str) -> None:
        addr = self._intern_pstr(text)
        # Push the string's start address (length word).
        self.compile_int_literal(addr)

    def _intern_pstr(self, text: str) -> int:
        addr = DATA_SEGMENT_START + len(self.data)
        self.data.append(len(text))
        for ch in text:
            self.data.append(ord(ch))
        return addr

    def compile_variable(self, name: str) -> None:
        depth = self.find_slot_depth(name)
        if depth is not None:
            if depth == 0:
                self.emit(Opcode.DUP, source=f"DUP   ; ref {name} (TOS)")
            elif depth == 1:
                self.emit(Opcode.OVER, source=f"OVER  ; ref {name} (NOS)")
            else:
                self.emit(Opcode.PICK, depth, source=f"PICK {depth}  ; ref {name}")
            self.push_slot()
            return
        if name in self.globals_:
            addr = self.globals_[name]
            self.emit(Opcode.LOAD, addr, source=f"LOAD 0x{addr:04X}  ; global {name}")
            self.push_slot()
            return
        raise TranslationError(f"undefined variable: {name}")

    def compile_list(self, form: list[Form]) -> None:
        if not form:
            raise TranslationError("empty form '()'")
        head = form[0]
        if not isinstance(head, Symbol):
            raise TranslationError(f"head must be a symbol, got {head!r}")
        name = head.name
        args = form[1:]

        if name in PRIMITIVE_BINARY:
            self._compile_binary(PRIMITIVE_BINARY[name], args, name)
            return
        if name in PRIMITIVE_UNARY:
            self._compile_unary(PRIMITIVE_UNARY[name], args, name)
            return
        if name in {"=", "<", ">"}:
            self._compile_compare(name, args)
            return
        if name == "setq":
            self._compile_setq(args)
            return
        if name == "if":
            self._compile_if(args)
            return
        if name == "progn":
            self._compile_progn(args)
            return
        if name == "let":
            self._compile_let(args)
            return
        if name == "halt":
            self._compile_halt(args)
            return
        if name == "print-char":
            self._compile_io_out(args, port=1, source_name="print-char")
            return
        if name == "print-int":
            self._compile_io_out(args, port=2, source_name="print-int")
            return
        if name == "read-char":
            self._compile_read_char(args)
            return
        if name == "load":
            self._compile_load(args)
            return
        if name == "store":
            self._compile_store(args)
            return
        if name in self.functions:
            self._compile_call(name, args)
            return
        raise TranslationError(f"unknown function or special form: {name}")

    # ----- specific compilers ----
    def _compile_binary(self, opcode: Opcode, args: list[Form], name: str) -> None:
        if len(args) != 2:
            raise TranslationError(f"({name} a b) takes 2 args, got {len(args)}")
        self.compile_expr(args[0])
        self.compile_expr(args[1])
        self.emit(opcode, source=opcode.name)
        # binary op pops 2, pushes 1
        self.pop_slot()
        # net: stack height -1

    def _compile_unary(self, opcode: Opcode, args: list[Form], name: str) -> None:
        if len(args) != 1:
            raise TranslationError(f"({name} a) takes 1 arg")
        self.compile_expr(args[0])
        self.emit(opcode, source=opcode.name)
        # stack height unchanged

    def _compile_compare(self, name: str, args: list[Form]) -> None:
        if len(args) != 2:
            raise TranslationError(f"({name} a b) takes 2 args")
        self.compile_expr(args[0])
        self.compile_expr(args[1])
        self.emit(Opcode.CMP, source=f"CMP   ; {name}")
        # CMP pops both operands, sets flags.
        self.pop_slot()
        self.pop_slot()

        true_label = self.gen_label("cmp_true")
        end_label = self.gen_label("cmp_end")
        branch_op = {"=": Opcode.JZ, "<": Opcode.JS, ">": None}[name]
        if name == ">":
            # a > b  <=>  a - b > 0  <=>  !Z and !N
            # Emit: if Z -> false; if N -> false; else true.
            false_first = self.gen_label("cmp_false")
            self.emit(Opcode.JZ, operand_label=false_first, source=f"JZ {false_first}")
            self.emit(Opcode.JS, operand_label=false_first, source=f"JS {false_first}")
            self.emit(Opcode.PUSH, 1, source="PUSH 1  ; >: true")
            self.emit(Opcode.JMP, operand_label=end_label, source=f"JMP {end_label}")
            self.attach_label(false_first)
            self.emit(Opcode.PUSH, 0, source="PUSH 0  ; >: false")
            self.attach_label(end_label)
        else:
            assert branch_op is not None
            self.emit(branch_op, operand_label=true_label, source=f"{branch_op.name} {true_label}")
            self.emit(Opcode.PUSH, 0, source="PUSH 0  ; cmp: false")
            self.emit(Opcode.JMP, operand_label=end_label, source=f"JMP {end_label}")
            self.attach_label(true_label)
            self.emit(Opcode.PUSH, 1, source="PUSH 1  ; cmp: true")
            self.attach_label(end_label)
        self.push_slot()

    def _compile_setq(self, args: list[Form]) -> None:
        if len(args) != 2:
            raise TranslationError("(setq name value) takes 2 args")
        name = _expect_symbol(args[0])
        if name in self.stack:
            raise TranslationError(f"setq on local '{name}' not supported; only globals")
        addr = self.globals_.setdefault(name, self.alloc_data_word())
        self.compile_expr(args[1])
        self.emit(Opcode.DUP, source=f"DUP   ; keep {name} as expr value")
        self.push_slot()
        self.emit(Opcode.STORE, addr, source=f"STORE 0x{addr:04X}  ; setq {name}")
        self.pop_slot()

    def _compile_if(self, args: list[Form]) -> None:
        if len(args) not in (2, 3):
            raise TranslationError("(if cond then [else])")
        self.compile_expr(args[0])
        # Cond on top: 0 = false. We need JZ to else branch; but JZ tests
        # the Z flag, not the stack value. Materialised values used CMP-based
        # construction that already set Z=(value == 0)? No — last op is PUSH
        # which doesn't touch flags. So compare to 0 explicitly.
        self.emit(Opcode.PUSH, 0, source="PUSH 0  ; if-cond cmp")
        self.push_slot()
        self.emit(Opcode.CMP, source="CMP   ; if-cond -> Z")
        self.pop_slot()
        self.pop_slot()
        else_label = self.gen_label("if_else")
        end_label = self.gen_label("if_end")
        self.emit(Opcode.JZ, operand_label=else_label, source=f"JZ {else_label}")
        # Then branch: result on TOS at end.
        self.compile_expr(args[1])
        self.emit(Opcode.JMP, operand_label=end_label, source=f"JMP {end_label}")
        # Reset stack height for parallel compile of else branch.
        self.pop_slot()
        self.attach_label(else_label)
        if len(args) == 3:
            self.compile_expr(args[2])
        else:
            self.emit(Opcode.PUSH, 0, source="PUSH 0  ; if without else -> 0")
            self.push_slot()
        self.attach_label(end_label)

    def _compile_progn(self, args: list[Form]) -> None:
        if not args:
            self.emit(Opcode.PUSH, 0, source="PUSH 0  ; empty progn")
            self.push_slot()
            return
        for i, expr in enumerate(args):
            self.compile_expr(expr)
            if i < len(args) - 1:
                self.emit(Opcode.DROP, source="DROP  ; progn non-final value")
                self.pop_slot()

    def _compile_let(self, args: list[Form]) -> None:
        if len(args) < 2:
            raise TranslationError("(let ((x e) ...) body...)")
        bindings = args[0]
        if not isinstance(bindings, list):
            raise TranslationError("let bindings must be a list")
        names: list[str] = []
        for b in bindings:
            if not isinstance(b, list) or len(b) != 2:
                raise TranslationError("each let binding is (name expr)")
            n = _expect_symbol(b[0])
            self.compile_expr(b[1])
            # Mark the slot as bound to `n` (was anonymous).
            self.stack[-1] = n
            names.append(n)
        body = args[1:]
        for i, expr in enumerate(body):
            self.compile_expr(expr)
            if i < len(body) - 1:
                self.emit(Opcode.DROP, source="DROP  ; let body non-final")
                self.pop_slot()
        # Result is on top, with each named slot just below in reverse order.
        # NIP each binding (SWAP DROP) so only the result remains.
        for n in names:
            self.emit(Opcode.SWAP, source=f"SWAP  ; bury result over '{n}'")
            self.emit(Opcode.DROP, source=f"DROP  ; remove let-binding '{n}'")
            # SWAP keeps height; DROP removes one slot — namely the named one
            # which is now at position [-2] before the DROP collapses it.
            del self.stack[-2]

    def _compile_halt(self, args: list[Form]) -> None:
        if args:
            raise TranslationError("(halt) takes no args")
        self.emit(Opcode.HALT, source="HALT")
        # Value: by convention 0 (never reached, but keep stack balanced).
        self.push_slot()

    def _compile_io_out(self, args: list[Form], port: int, source_name: str) -> None:
        if len(args) != 1:
            raise TranslationError(f"({source_name} v) takes 1 arg")
        self.compile_expr(args[0])
        # Keep a copy as the expression value, since OUTPUT pops one.
        self.emit(Opcode.DUP, source="DUP   ; keep value as expr result")
        self.push_slot()
        self.emit(Opcode.OUTPUT, port, source=f"OUTPUT {port}  ; {source_name}")
        self.pop_slot()

    def _compile_read_char(self, args: list[Form]) -> None:
        if args:
            raise TranslationError("(read-char) takes no args")
        self.emit(Opcode.INPUT, 0, source="INPUT 0  ; read-char")
        self.push_slot()

    def _compile_load(self, args: list[Form]) -> None:
        if len(args) != 1:
            raise TranslationError("(load addr) takes 1 arg")
        if isinstance(args[0], int):
            self.emit(Opcode.LOAD, args[0], source=f"LOAD 0x{args[0]:04X}")
            self.push_slot()
            return
        # General: compute address on stack, then LOADI.
        self.compile_expr(args[0])
        self.emit(Opcode.LOADI, source="LOADI ; indirect load")

    def _compile_store(self, args: list[Form]) -> None:
        if len(args) != 2:
            raise TranslationError("(store addr value) takes 2 args")
        self.compile_expr(args[1])
        if isinstance(args[0], int):
            self.emit(Opcode.DUP, source="DUP   ; keep value")
            self.push_slot()
            self.emit(Opcode.STORE, args[0], source=f"STORE 0x{args[0]:04X}")
            self.pop_slot()
            return
        raise TranslationError("(store <dynamic addr> value) not yet supported")

    def _compile_call(self, name: str, args: list[Form]) -> None:
        label = self.functions[name]
        # Push args left-to-right so the last is on TOS — matches how the
        # function's params were registered in defun (list order).
        for a in args:
            self.compile_expr(a)
        self.emit(Opcode.CALL, operand_label=label, source=f"CALL {label}  ; ({name} ...)")
        # Function pops its args and pushes one return value.
        for _ in args:
            self.pop_slot()
        self.push_slot()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_call(form: Form, head: str) -> bool:
    return isinstance(form, list) and len(form) > 0 and isinstance(form[0], Symbol) and form[0].name == head


def _expect_symbol(form: Form) -> str:
    if not isinstance(form, Symbol):
        raise TranslationError(f"expected symbol, got {form!r}")
    return form.name


# ---------------------------------------------------------------------------
# Linker: place code, resolve labels, build image
# ---------------------------------------------------------------------------


@dataclass
class LinkedProgram:
    image: list[int]  # word values at addresses 0..N-1
    listing: str


def link(compiler: Compiler, code_base: int = DEFAULT_CODE_BASE) -> LinkedProgram:
    # Code placement: assign every CodeItem an address.
    code_address = code_base
    label_addr: dict[str, int] = {}
    item_addr: list[int] = []
    for item in compiler.code:
        item_addr.append(code_address)
        if item.label is not None:
            if item.label in label_addr:
                raise TranslationError(f"duplicate label: {item.label}")
            label_addr[item.label] = code_address
        code_address += 1

    # Resolve operand_label to actual addresses.
    for item in compiler.code:
        if item.operand_label is not None:
            if item.operand_label not in label_addr:
                raise TranslationError(f"unresolved label: {item.operand_label}")
            item.operand = label_addr[item.operand_label]

    # Build the memory image: reset vector at 0, data segment, code segment.
    image_size = code_base + len(compiler.code)
    image = [0] * image_size
    if "_start" not in label_addr:
        raise TranslationError("internal: missing _start label")
    image[0] = encode(Instr(opcode=Opcode.JMP, operand=label_addr["_start"]))
    for i, word in enumerate(compiler.data):
        image[DATA_SEGMENT_START + i] = word
    for i, item in enumerate(compiler.code):
        image[code_base + i] = encode(Instr(opcode=item.opcode, operand=item.operand))

    # Listing: reset vector, then data section, then code section.
    lines: list[str] = []
    reset_word = image[0]
    lines.append(f"0000  {reset_word:08X}  JMP   0x{label_addr['_start']:04X}      ; reset vector")
    lines.append("")
    lines.append("; data segment")
    for i, word in enumerate(compiler.data):
        addr = DATA_SEGMENT_START + i
        lines.append(f"{addr:04X}  {word & 0xFFFFFFFF:08X}  DATA  {word}")
    lines.append("")
    lines.append("; code segment")
    for item, addr in zip(compiler.code, item_addr, strict=True):
        instr = Instr(item.opcode, item.operand)
        word = encode(instr)
        label_str = f"{item.label}:" if item.label else ""
        lines.append(f"{addr:04X}  {word:08X}  {mnemonic(instr):<20s}  {label_str} {item.source}")
    listing = "\n".join(lines) + "\n"
    return LinkedProgram(image=image, listing=listing)


def image_to_bytes(image: list[int]) -> bytes:
    return b"".join(int(w & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False) for w in image)


# ---------------------------------------------------------------------------
# Public API and CLI
# ---------------------------------------------------------------------------


def translate(source: str) -> tuple[bytes, str]:
    tokens = tokenize(source)
    program = parse(tokens)
    compiler = Compiler()
    compiler.compile_program(program)
    linked = link(compiler)
    return image_to_bytes(linked.image), linked.listing


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
