# Csa-Lab4

**Студент:** Сушко Александр, P3230 (заменить на реальную группу при необходимости).

**Вариант:** `lisp | stack | neum | mc | tick | binary | stream | port | pstr | alg1`

Лабораторная работа №4: моделирование процессора с архитектурой стека, фон-Неймановской организацией памяти, микропрограммным управлением, тактовой точностью, бинарным машинным кодом, потоковым port-mapped вводом-выводом, Pascal-строками. Целевой алгоритм — Euler Problem 4 (наибольшее палиндромное произведение двух трёхзначных чисел).

## Запуск

```bash
pip install -e ".[dev]"

translator examples/hello.lisp out/hello.bin --listing out/hello.lst
machine     out/hello.bin    input.txt        --log out/hello.log
```

## CI

GitHub Actions: ruff (lint + format), mypy strict, pytest с покрытием. См. `.github/workflows/ci.yml`.

## Структура

```
csa_lab4/
  isa.py          — опкоды, кодирование 32-битных команд
  microcode.py    — сигналы, селекторы, ROM микрокода
  translator.py   — лексер/парсер S-exp, кодогенерация
  machine.py      — DataPath + microcoded ControlUnit + I/O
tests/            — unit + golden тесты
examples/         — программы на нашем lisp-диалекте
```

Полный отчёт — будет дополнен по мере реализации.
