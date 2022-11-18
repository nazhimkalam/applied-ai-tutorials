

(define (problem blocks)(:domain blocksworld)

; Objects that are used in this problem
(:objects
    red green blue yellow brown pink - block
    )

(:init
    ;tower
    (ontable red) ; Block red
    (on green red) ; Block green
    (on blue green)(clear blue) ; Block blue
    ;tower
    (ontable yellow) ; Block yellow
    (on brown yellow) ; Block brown
    (on pink brown)(clear pink) ; Block pink

    (handempty) ; the robot hand is empty
)

(:goal (and
    (on red brown)
    (on green red)
    (holding yellow) ; the robot is holding yellow
))
)