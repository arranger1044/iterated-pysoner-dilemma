
@startuml
start
repeat
:simulate pairwise IPD games;
:aggregate scores by **type**;
:compute likelihoods on scores;
:create new generation;
repeat while (another generation?)
stop
@enduml 