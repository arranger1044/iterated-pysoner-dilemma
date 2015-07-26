
@startuml
class IPDPlayer{
    +id
    -_type
    -_ingame_id
    --
    +act(game, iter)
}
class PERPlayer{
    -_pattern
    -_pattern_length
    -_pos
}
PERPlayer -r-|> IPDPlayer
AllCPlayer -r-|> PERPlayer
AllDPlayer -r-|> PERPlayer
@enduml