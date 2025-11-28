from ga_fuzzy_aimbot_controller import GeneticFuzzyAimbotController

best = GeneticFuzzyAimbotController.optimize(
    ga=GeneticFuzzyAimbotController.GAParams(
        generations=2,   # small
        population=6,    # small
        elites=1,
        mutation_prob=0.25,
        crossover_prob=0.9,
    ),
    eval_games=1,       # tiny eval
    seed=0
)
print("Best params:", best)
