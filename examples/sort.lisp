; Sort a list of integers read from stdin.
;
; Input format (length-prefixed, analogous to pstr): first integer is the
; count N, then N integers, each followed by a space; the final integer is
; followed by a newline. Whitespace is the only allowed separator.
;
; Example input :  "5 3 1 4 1 5\n"
; Example output:  "1 1 3 4 5\n"

(setq buf (buffer-of 32))

(defun read-digits (acc)
  (let ((c (read-char)))
    (if (= c 32)            ; space  -> end of number
        acc
        (if (= c 10)        ; newline -> end of number
            acc
            (read-digits (+ (* acc 10) (- c 48)))))))

(defun read-int () (read-digits 0))

(defun fill-buf (ptr i n)
  (if (= i n)
      0
      (progn
        (store-at (+ ptr i) (read-int))
        (fill-buf ptr (+ i 1) n))))

(defun bubble-pass (ptr i n swapped)
  (if (= (+ i 1) n)
      swapped
      (let ((a (load (+ ptr i)))
            (b (load (+ ptr (+ i 1)))))
        (if (> a b)
            (progn
              (store-at (+ ptr i) b)
              (store-at (+ ptr (+ i 1)) a)
              (bubble-pass ptr (+ i 1) n 1))
            (bubble-pass ptr (+ i 1) n swapped)))))

(defun bubble-sort (ptr n)
  (if (= 1 (bubble-pass ptr 0 n 0))
      (bubble-sort ptr n)
      0))

(defun print-loop (ptr i n)
  (if (= i n)
      0
      (progn
        (print-int (load (+ ptr i)))
        (print-char 32)
        (print-loop ptr (+ i 1) n))))

(setq n (read-int))
(fill-buf buf 0 n)
(bubble-sort buf n)
(print-loop buf 0 n)
(print-char 10)
(halt)
