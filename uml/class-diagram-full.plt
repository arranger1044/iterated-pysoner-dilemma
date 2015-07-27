
@startuml
class IPDPlayer{
    +id
    -_type
    -_ingame_id
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
class IPDEvolutionarySimulation{
    -_player_types
    -_payoffs
    -_population
    -_n_generations
    -_n_iters
    -_noise
    -_average
    --
    {static} + create_fixed_generation(player_types, population_per_type)
    {static} + create_generation(player_types, player_likelihoods, population)
    {static} + plot_simulation(frame)
    + simulate(first_generation, n_generations, printing)
}
IPDPlayer "2" *-- "1..*" IPDGame
IPDGame "1..*" -- "1" IPDPairwiseCompetition
IPDPairwiseCompetition "1..*" --* "2..*" IPDPlayer
IPDPairwiseCompetition "1..*" -r- "2..*" IPDEvolutionarySimulation
IPDEvolutionarySimulation "1..*" -r- "1..*"  IPDPlayer
@enduml 