;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-

#|
=========================================================
Module: load-eliza.lisp: 
Description: a load for three eliza modules.
bugs to vladimir dot kulyukin via canvas
=========================================================
|#

;;; change this parameter as needed.
(defparameter *eliza-path* 
  "/mnt/c/Users/Bryson\ M/Documents/USU/Classes/Intelligent\ Systems/hw08/")

;;; the files that comprise ELIZA.
(defparameter *eliza-files* 
  '("auxfuns.lisp" "eliza.lisp")) 

(defun load-eliza-aux (path files)
  (mapc #'(lambda (file)
	    (load (concatenate 'string path file)))
	files))

;;; load ELIZA
(defun load-eliza ()
  (load-eliza-aux *eliza-path* *eliza-files*))

	
