(define (problem logistics-4-2)
(:domain logistics)
; Logistic domain problem with 3 packages to move from one city to another city

(:objects apn1 apt2 pos2 apt1 pos1 cit2 cit1 tru2 tru1 obj21 obj13 obj12 obj11 )
(:init 
        (package obj11) (package obj12) (package obj21) 
        (truck tru1) (truck tru2) 
        (city cit1) (city cit2)
        (location pos1) (location apt1) (location pos2) (location apt2) 
        (airport apt1) (airport apt2) (airplane apn1) 
        (at apn1 apt1) (at tru1 pos1) (at obj11 pos1) (at obj12 pos1) (at tru2 pos2) (at obj21 pos2) 
        
        (in-city pos1 cit1) (in-city apt1 cit1) (in-city pos2 cit2)(in-city apt2 cit2))
        
(:goal (and 
            (at obj21 apt1) 
            (at obj11 pos2) 
            (at obj12 pos1)))
)
