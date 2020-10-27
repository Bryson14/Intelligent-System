;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-

#|
=========================================================
Module: eliza.lisp: 
Description: A version of ELIZA that takes inputs without 
paretheses around them unlike eliza1.lisp.
Bugs to vladimir dot kulyukin via canvas
=========================================================
|#

#|
Best conversation 
ELIZA>  i love my dad
DOES YOUR DAD KNOW ABOUT
ELIZA> i love my dad for his brain
I AM SORRY BUT WHAT DOES YOUR DAD HAVE TO DO WITH THIS?
ELIZA> i love my dad for his brain
DOES YOUR DAD KNOW ABOUT FOR HIS BRAIN
ELIZA> did you know i hate you
WHY DO YOU HATE I
ELIZA> i am angry with you
ITS OK TO FELL ANGRY
ELIZA> you are a computer
WHY DO YOU MENTION COMPUTERS?
ELIZA> you are a computer
WHY DO YOU MENTION COMPUTERS?
ELIZA> im tired of this crap
DOES THIS CRAP MAKE YOU OFTEN TIRED?
ELIZA> hey hey whats up
HOW IS IT HANING?
ELIZA> lol hey
I FEEL LIKE WE ARE FRIENDS
ELIZA> i need some exersise

ELIZA> i need someting

ELIZA> i need exercise
DO YOU YOU NEED EXERCISE TO FELL
ELIZA> i need exercise to be better
DO YOU YOU NEED EXERCISE TO FELL TO BE BETTER
ELIZA> are you hungry
DO YOU LOVED ONE KNOW ABOUT ARE I
ELIZA> why isn't this product an apple
YOU NEED SOME MORE DRIP
ELIZA> why isn't this product an apple product
YOU NEED SOME MORE DRIP
ELIZA> why isn't this product an apple product
YOU NEED SOME MORE DRIP
ELIZA>

ELIZA> why isn't this product an apple product
YOU NEED SOME MORE DRIP
ELIZA> why isn't this product an apple product
YOU NEED SOME MORE DRIP
ELIZA> why isn't this product an apple product
I WAS DOING PRODUCT YEARS AGO BEFORE APPLE EVEN WAS BORN
ELIZA> why isn't this product an apple product
YOU NEED SOME MORE DRIP
ELIZA> big tech scares me
DOES ADVANCEMENT SCARE YOU?
ELIZA> i see a little sillouette of a man
SCARAMOUCHE SCARAMOUCHE WILL YOU DO THE FANDANGO? THUNDERBOLT AND LIGHTNING VERY VERY FRIGHTENING ME.
ELIZA> help
WHY DO YOU NEED HELP
ELIZA> you are stupid
DOES YOUR STUPIDITY SCARE YOU OR DO YOU JUST NOT GET IT?
ELIZA>

|#

;;; ==============================

(defun rule-pattern (rule) (first rule))
(defun rule-responses (rule) (rest rule))

(defun read-line-no-punct ()
  "Read an input line, ignoring punctuation."
  (read-from-string
    (concatenate 'string "(" (substitute-if #\space #'punctuation-p
                                            (read-line))
                 ")")))

(defun punctuation-p (char) (find char ".,;:`!?#-()\\\""))

;;; ==============================

(defun use-eliza-rules (input)
  "Find some rule with which to transform the input."
  (some #'(lambda (rule)
            (let ((result (pat-match (rule-pattern rule) input)))
              (if (not (eq result fail))
                  (sublis (switch-viewpoint result)
                          (random-elt (rule-responses rule))))))
        *eliza-rules*))

(defun switch-viewpoint (words)
  "Change I to you and vice versa, and so on."
  (sublis '((i . you) (you . i) (me . you) (am . are) (my . your) (your . my))
          words))

(defparameter *good-byes* '((good bye) (see you) (see you later) (so long)))

(defun eliza ()
  "Respond to user input using pattern matching rules."
  (loop
    (print 'eliza>)
    (let* ((input (read-line-no-punct))
           (response (flatten (use-eliza-rules input))))
      (print-with-spaces response)
      (if (member response *good-byes* :test #'equal)
	  (RETURN))))
  (values))

(defun print-with-spaces (list)
  (mapc #'(lambda (x) (prin1 x) (princ " ")) list))

(defun print-with-spaces (list)
  (format t "~{~a ~}" list))

;;; ==============================

(defparameter *eliza-rules*
  '(
    ;;; rule 1
    (((?* ?x) hello (?* ?y))      
    (How do you do.  Please state your problem.))

    ;;; rule 2
    (((?* ?x) computer (?* ?y))
     (Do computers worry you?)
     (What do you think about machines?)
     (Why do you mention computers?)
     (What do you think machines have to do with your problem?))

    ;;; rule 3
    (((?* ?x) name (?* ?y))
     (I am not interested in names))

    ;;; rule 4
    (((?* ?x) sorry (?* ?y))
     (Please don't apologize)
     (Apologies are not necessary)
     (What feelings do you have when you apologize))

    ;;; rule 5
    (((?* ?x) remember (?* ?y)) 
     (Do you often think of ?y)
     (Does thinking of ?y bring anything else to mind?)
     (What else do you remember)
     (Why do you recall ?y right now?)
     (What in the present situation reminds you of ?y)
     (What is the connection between me and ?y))

    ;;; rule 6
    (((?* ?x) good bye (?* ?y))
     (good bye))

    ;;; rule 7
    ;;;(((?* ?x) so long (?* ?y))
    ;;; (good bye)
    ;;; (bye)
    ;;; (see you)
    ;;; (see you later))

    ;;; ========== your rules begin
    ;;; rule 1
    (((?* ?x) hate (?* ?y))
     (Why do you hate ?y)
	 (Does hating ?y make you feel better?))

    ;;; rule 2
    (((?* ?x) hurts me (?* ?y))
     (What did ?x do to hurt you?)
	 (Did ?x mean to hurt you?))
	 
    ;;; rule 3
    (((?* ?x) angry (?* ?y))
     (Its ok to fell angry)
	 (What is the best thing to do when you are angry))

    ;;; rule 4
    (((?* ?x) a computer (?* ?y))
     (?x you a computer?)
	 (How do you know you are not a machine?))

    ;;; rule 5
    (((?* ?x) gucci (?* ?y))
     (?y make me feel gucci too)
	 (hmm lets coat you is some gucci drizzle!))

    ;;; rule 6
    (((?* ?x) tired of (?* ?y))
     (Does ?y make you often tired?)
	 (Sometimes everyone get tired of their own version of ?y))

    ;;; rule 7
    (((?* ?x) dad (?* ?y))
     (I am sorry but what does your dad have to do with this?)
	 (Does your dad know about ?y))

    ;;; rule 8
    (((?* ?x) hey (?* ?y))
     (What is up?)
	 (How is it haning?)
	 (I feel like we are friends))

    ;;; rule 9
    (((?* ?x) exercise (?* ?y))
     (Do you ?x exercise to fell ?y)
	 (You need ?y to exercise? It seems so to me.))

    ;;; rule 10
    (((?* ?x) hungry (?* ?y))
     (What about ?x makes you hungry?)
	 (Do you loved one know about ?x))

    ;;; rule 11
    (((?* ?x) apple (?* ?y))
     (You need some more drip)
	 (I was doing ?y years ago before apple even was born))

    ;;; rule 12
    (((?* ?x) tech(?* ?y))
     (Does advancement scare you?)
	 (What does tech have to do with this?)
	 (This is my favorite ?y ever))

    ;;; rule 13
    (((?* ?x) potato (?* ?y))
     (Boil em Mash em Stick em in a stew))

    ;;; rule 14
    (((?* ?x) sing (?* ?y))
     (It is ok to sing ?y)
	 (if you like to sing but badly do it in the shower))

    ;;; rule 15
    (((?* ?x) golf (?* ?y))
     (Lets go golfing sometime)
	 (?x is the best way to golf))

    ;;; rule 16
    (((?* ?x) mom (?* ?y))
     (Moms can be the best or the worst)
	 (What does thinking of your mom make you feel?))

    ;;; rule 17
    (((?* ?x) I see a little sillouette of a man (?* ?y))
     (Scaramouche Scaramouche will you do the Fandango? Thunderbolt and lightning very very frightening me.)
	 (Galileo Galileo))

    ;;; rule 18
    (((?* ?x) help (?* ?y))
     (Why do you need help ?y)
	 (What incentive ?x to help ?y))

    ;;; rule 19
    (((?* ?x) stupid (?* ?y))
     (Stupid is as stupid does)
	 (Does your stupidity scare you or do you just not get it?))

    ;;; rule 20
    (((?* ?x) helpless (?* ?y))
     (?x helpless is a answer that denies you of your own responsibility)
	 (is the ?y ok?))	 
    ;;; ========== your rules end

   ))

;;; ==============================

