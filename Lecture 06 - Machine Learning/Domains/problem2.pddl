(define (problem blocks)(:domain blocksworld)

(:objects
    A B C  - block
    )

(:init
    ;tower
    (ontable A) ; Block A on table
    (on C A) ; Block C on A
    (clear C) ; Block C clear
    (ontable B)
    (clear B)
    (handempty)
)

(:goal (and
    (on B C)
    (on A B)
    (handempty)
))
)