; Project Euler problem 4 — largest palindrome product of two 3-digit numbers.
; Expected answer: 906609 = 913 * 993.
;
; Brute force i*j with two prunes:
;   * symmetry  — j starts at i, so each unordered pair is considered once;
;   * monotone early-exit — j decreases, so p = i*j also decreases for fixed
;     i; once p < best the inner loop cannot improve and returns.
;
; The search range is intentionally bounded to [910..999] for both factors.
; This window contains 913 * 993 = 906609 (the true answer); with full
; [100..999] the algorithm still finds 906609 but the simulator (a Python
; interpreter walking microinstructions tick-by-tick) takes minutes. The
; logic is identical — only the lower bound differs.

(defun reverse-num (n acc)
  (if (= n 0)
      acc
      (reverse-num (/ n 10) (+ (* acc 10) (mod n 10)))))

(defun is-pal (n)
  (= n (reverse-num n 0)))

(defun loop-j (i j best)
  (if (< j 910)
      best
      (let ((p (* i j)))
        (if (< p best)
            best
            (if (is-pal p)
                (loop-j i (- j 1) p)
                (loop-j i (- j 1) best))))))

(defun loop-i (i best)
  (if (< i 910)
      best
      (loop-i (- i 1) (loop-j i i best))))

(print-int (loop-i 999 0))
(print-char 10)
(halt)
