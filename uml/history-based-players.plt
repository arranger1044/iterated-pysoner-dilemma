
@startuml
class IPDPlayer{
    +id
    -_type
    -_ingame_id
    --
    +act(game, iter)
}
class HBPlayer{
    -_history_length
    -_default_policy
    --
    -_get_history(iter)
}
class GRIMPlayer{
    -_defecting
}
class PHBPlayer{
    -_alpha
    --
    -_predict_next(history)
}
HBPlayer -r-|> IPDPlayer
GRIMPlayer --|> IPDPlayer
SMPlauer -r-|> HBPlayer
HMPlayer -r-|> HBPlayer
PHBPlayer --|> HBPlayer
@enduml