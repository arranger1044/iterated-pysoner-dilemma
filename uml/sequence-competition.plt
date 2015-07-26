
@startuml
IPDPairwiseCompetition -> IPDPlayer1: select
IPDPairwiseCompetition -> IPDPlayer2: select
IPDPairwiseCompetition -> IPDGame: __init__()
IPDGame -> IPDPlayer1: set_ingame_id(0)
IPDGame -> IPDPlayer2: set_ingame_id(1)    
    loop i:n_iters
        IPDGame -> IPDPlayer1: act(self, i)
        IPDPlayer1 --> IPDGame: action
        IPDGame -> IPDPlayer2: act(self, i)
        IPDPlayer2 --> IPDGame: action
        IPDGame -> IPDGame: _get_payoff(actions)
    end
    IPDGame --> IPDPairwiseCompetition: actions, rewards
@enduml 