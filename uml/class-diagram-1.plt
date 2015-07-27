
@startuml
class IPDPlayer{
    +id
    -_type
    -_ingame_id
    -_noise
    --
    +act(game, iter)
}
class IPDGame{
    -_payoffs
    -_players
    +actions
    +rewards
    --
    +payoffs()
    +get_actions(start, end)
    +simulate(iters)
}
class IPDPairwiseCompetition{
    -_payoffs
    -_players
    --
    +simulate(iters)
}
IPDPlayer "2" *-- "1..*" IPDGame
IPDGame "1..*" -- "1" IPDPairwiseCompetition
IPDPairwiseCompetition "1..*" --* "2..*" IPDPlayer
@enduml 