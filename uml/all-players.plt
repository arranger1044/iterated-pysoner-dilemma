
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

M1Player --|> IPDPlayer
TFTPlayer --|> M1Player
WSLSPlayer --|> M1Player

class PERPlayer{
    -_pattern
    -_pattern_length
    -_pos
}
PERPlayer --|> IPDPlayer
PCCDPlayer --|> PERPlayer
PDDCPlayer --|> PERPlayer

class HBPlayer{
    -_history_length
    -_default_policy
}
class GRIMPlayer{
    -_defecting
}
class PHBPlayer{
    -_alpha
    --
    -_predict_next(history)
}
HBPlayer --|> IPDPlayer
GRIMPlayer --|> IPDPlayer
SMPlauer --|> HBPlayer
HMPlayer --|> HBPlayer
PHBPlayer --|> HBPlayer

class GTFTPlayer{
    -_theta
}
class ATFTPlayer{
    -_theta
    -_world
}

STFTPlayer --|> TFTPlayer
GTFTPlayer --|> TFTPlayer
RTFTPlayer --|> TFTPlayer
ATFTPlayer --|> TFTPlayer

AllCPlayer --|> PERPlayer
AllDPlayer --|> PERPlayer

class RANDPlayer{
    -_theta
}
RANDPlayer --|> IPDPlayer
@enduml