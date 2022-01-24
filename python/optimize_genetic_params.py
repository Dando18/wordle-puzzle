''' A script for optimizing the genetic guessing policy.
    author: Daniel Nichols
    date: January 2022
'''
from functools import partial

from hyperopt import fmin, tpe, hp, space_eval

from wordle import Wordle
from simulator import Simulator
from wordle_guess_policy import GeneticGuessPolicy
from utility import read_word_list, filter_word_list


MAX_SIM_ITER = 50
MAX_OBJ_EVALS = 200
MP_PROCS = 4

# from training over 200 iterations with max 20 games per iteration
# achieved 6.95 on full word list
OPTIMAL = {
    'crossover_prob': 0.31205820872312595, 
    'diversify': True, 
    'fitness_const': 7.0, 
    'invert_prob': 0.025779667955558305, 
    'max_eligible_size': 500.0, 
    'max_generations': 50, 
    'mutate_prob': 0.09457252541449512, 
    'permute_prob': 0.06168892887298874, 
    'population_size': 1000.0, 
    'tournament_size': 90.0
    }

def objective(simulator, kwargs):
    simulator.policy_ = GeneticGuessPolicy(first_guess='scare', **kwargs)
    return simulator.simulate_games_multiprocessing(num_games=MAX_SIM_ITER, 
                                        show_progress=False, procs=MP_PROCS)


def main():

    # read in word list and preprocess (about 16000 5 letter words)
    #word_list = read_word_list('../english-words/words_dictionary.json')
    word_list = read_word_list('../word-list-solutions.txt')
    #word_list = filter_word_list(word_list, length=5)

    game = Wordle(word_list, max_iter=100)
    policy = GeneticGuessPolicy()
    sim = Simulator(game, policy)

    obj = partial(objective, sim)
    search_space = {
        'population_size': hp.quniform('population_size', 100, 2315, 200),
        'max_generations': hp.choice('max_generations', [20, 50, 100, 200]),
        'max_eligible_size': hp.quniform('max_eligible_size', 50, 2315, 50),
        'tournament_size': hp.quniform('tournament_size', 10, 200, 10),
        'crossover_prob': hp.uniform('crossover_prob', 0.1, 0.9),
        'mutate_prob': hp.uniform('mutate_prob', 0.01, 0.1),
        'permute_prob': hp.uniform('permute_prob', 0.01, 0.1),
        'invert_prob': hp.uniform('invert_prob', 0.01, 0.1),
        'fitness_const': hp.quniform('fitness_const', 0, 10, 1),
        'diversify': hp.choice('diversify', [True, False])
    }

    best = fmin(fn=obj,
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_OBJ_EVALS)

    print(best)
    print(space_eval(search_space, best))


if __name__ == '__main__':
    main()