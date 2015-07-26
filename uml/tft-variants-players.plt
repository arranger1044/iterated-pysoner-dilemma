
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
class GTFTPlayer{
    -_theta
}
class ATFTPlayer{
    -_theta
    -_world
}
M1Player -r-|> IPDPlayer
TFTPlayer -r-|> M1Player
STFTPlayer -r-|> TFTPlayer
GTFTPlayer --|> TFTPlayer
RTFTPlayer --|> TFTPlayer
ATFTPlayer --|> TFTPlayer
@enduml