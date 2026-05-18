; Echo input characters until the input buffer is exhausted, at which point
; the simulator halts (per the stream variant: empty buffer => HaltError).
(defun cat ()
  (progn
    (print-char (read-char))
    (cat)))
(cat)
