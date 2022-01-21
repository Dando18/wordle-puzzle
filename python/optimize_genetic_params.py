''' A script for optimizing
'''
from functools import partial

from hyperopt import fmin, tpe, hp

from wordle import Wordle
from wordle_simulator import Simulator
from wordle_guess_policy import GeneticGuessPolicy


MAX_SIM_ITER = 10
MAX_OBJ_EVALS = 200
MP_PROCS = 4


def read_word_list(fname):
    with open(fname, 'r') as fp:
        if fname.endswith('.json'):
            from json import load
            return list(load(fp).keys())
        else:
            return list(fp.readlines())


def filter_word_list(words, length=5):
    return list( filter(lambda x: len(x) == length, words) )


def objective(simulator, kwargs):
    simulator.policy_ = GeneticGuessPolicy(first_guess='cares', **kwargs)
    return simulator.simulate_games_multiprocessing(num_games=MAX_SIM_ITER, show_progress=False, procs=MP_PROCS)


def main():

    # read in word list and preprocess (about 16000 5 letter words)
    word_list = read_word_list('../english-words/words_dictionary.json')
    word_list = filter_word_list(word_list, length=5)

    game = Wordle(word_list, max_iter=100)
    policy = GeneticGuessPolicy()
    sim = Simulator(game, policy)

    obj = partial(objective, sim)
    search_space = {
        'population_size': hp.quniform('population_size', 100, 1000, 200),
        'max_generations': hp.choice('max_generations', [20, 50, 100, 200]),
        'max_eligible_size': hp.quniform('max_eligible_size', 50, 500, 50),
        'tournament_size': hp.quniform('tournament_size', 10, 100, 10),
        'crossover_prob': hp.uniform('crossover_prob', 0.3, 0.8),
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


if __name__ == '__main__':
    main()