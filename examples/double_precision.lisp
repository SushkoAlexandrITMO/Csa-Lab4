; Double-precision (62-bit) addition demonstrated on top of 32-bit machine
; words. We represent a 62-bit number as two 31-bit halves (hi, lo); each
; half lives in one signed 32-bit machine word with the sign bit kept
; clear. The carry from lo into hi is detected by observing the sign of the
; native 32-bit addition: lo1 + lo2 fits in [0, 2^32-2], and when the result
; >= 2^31 the native signed view becomes negative — exactly when a 31-bit
; carry has occurred.
;
; Demonstrated case:
;   A = (1, 2_000_000_000)
;   B = (1, 1_500_000_000)
;   A + B in true 62-bit arithmetic = 7_794_967_296
;                                   = 3 * 2^31 + 1_352_516_352
; Expected output:  hi=3 lo=1352516352

(setq pow31 2147483648)   ; bit pattern 0x80000000; ALU reads as -2^31

(setq ahi 1)
(setq alo 2000000000)
(setq bhi 1)
(setq blo 1500000000)

(setq lo-sum (+ alo blo))
(setq carry (if (< lo-sum 0) 1 0))
; If carry: native sum wrapped past 2^31. Adding pow31 (whose ALU value is
; -2^31) brings it back into the unsigned 0..2^31-1 range modulo 2^32.
(setq lo-final (if (< lo-sum 0) (+ lo-sum pow31) lo-sum))
(setq hi-final (+ ahi (+ bhi carry)))

(print-string "hi=")
(print-int hi-final)
(print-string " lo=")
(print-int lo-final)
(print-char 10)
(halt)
