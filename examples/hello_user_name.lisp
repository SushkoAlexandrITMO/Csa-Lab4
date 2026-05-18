; Interactive greeting:
;   prompt -> read name (terminated by \n) -> print "Hello, <name>!"
;
; read-string consumes characters until a newline (0x0A) into the pstr at
; buf, storing the length in buf's first word. The prelude implements both
; reading and printing.

(setq buf (buffer-of 32))

(print-string "What is your name?")
(print-char 10)
(read-string buf)
(print-string "Hello, ")
(print-string buf)
(print-char 33)   ; '!'
(print-char 10)
(halt)
