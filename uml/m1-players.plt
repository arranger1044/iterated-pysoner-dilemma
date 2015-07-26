
@startuml
class IPDPlayer{
    +id
    -_type
    -_ingame_id
    --
    +act(game, iter)
}
class M1Player{
    -_cond_probs
    -_default_policy
}
class TFTPlayer{
    -_cond_probs_0
    -_cond_probs_1
}
M1Player -r-|> IPDPlayer
TFTPlayer -r-|> M1Player
WSLSPlayer -r-|> M1Player
@enduml