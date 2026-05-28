; Double-precision (64-bit) addition using the hardware carry flag (C) and
; the ADC instruction. A 64-bit number is stored as two 32-bit machine words
; (hi, lo):
;
;   (+ alo blo)    -> ADD: low word; sets C on unsigned 32-bit overflow
;   (adc ahi bhi)  -> ADC: high word; folds in the carry from the low add
;
; Nothing between the two operations touches C (loads / setq / drop all leave
; the flag intact), so the carry propagates correctly from low to high.
;
; Demonstrated:
;   A = 7_294_967_296  = (1, 3_000_000_000)
;   B = 10_589_934_592 = (2, 2_000_000_000)
;   A + B = 17_884_901_888 = (4, 705_032_704)
; Expected output:  hi=4 lo=705032704

(setq ahi 1)
(setq alo 3000000000)
(setq bhi 2)
(setq blo 2000000000)

(setq lo (+ alo blo))      ; ADD sets carry
(setq hi (adc ahi bhi))    ; ADC consumes carry

(print-string "hi=")
(print-int hi)
(print-string " lo=")
(print-int lo)
(print-char 10)
(halt)
