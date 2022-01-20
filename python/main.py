''' Wordle simulator and/or AI
author: Daniel Nichols
date:  January 2021
'''
from argparse import ArgumentParser
from collections import Counter
import logging
import random

from wordle import Wordle
from wordle_simulator import Simulator
from wordle_guess_policy import RandomGuessPolicy, MinimaxGuessPolicy, ProbabalisticGreedyGuessPolicy, \
                                GeneticGuessPolicy

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', 
        type=str.upper, help='logging level')
    parser.add_argument('--word-list', type=str, default='../english-words/words_dictionary.json', 
        help='where to find english word list.')
    parser.add_argument('-l', '--length', type=int, default=5, help='what length word to use.')
    parser.add_argument('-g', '--guesses', default=None, help='how many guesses')
    parser.add_argument('-n', '--num-games', type=int, default=20, help='# of games to simulate')
    parser.add_argument('-p', '--policy', type=str.lower, choices=['random', 'smart_random', 'minimax', 'prob_greedy',
        'genetic'], default='random', help='guessing policy')
    parser.add_argument('--smart-first-guess', action='store_true', help='Use an optimal first guess.')
    parser.add_argument('--seed', type=int, default=-1, help='random seed. -1 for system time.')
    parser.add_argument('--multiprocessing', type=int, nargs='?', const=-1, help='use multiprocessing in simulation')
    return parser.parse_args()


def read_word_list(fname):
    with open(fname, 'r') as fp:
        if fname.endswith('.json'):
            from json import load
            return list(load(fp).keys())
        else:
            return list(fp.readlines())


def filter_word_list(words, length=5):
    return list( filter(lambda x: len(x) == length, words) )


def smart_first_guess(words):
    import numpy as np

    words_mat = list( map(lambda s: [c for c in s], words) )
    words_mat = np.array(words_mat).T

    hist = list( map(Counter, words_mat) )
    best_word = ''.join( map(lambda c: c.most_common(1)[0][0], hist) )

    if best_word in words:
        return best_word
    
    rankings = [{pair[0]: rank for rank, pair in enumerate(c.most_common())} for c in hist]

    best_score = sum([rankings[idx][letter] for idx, letter in enumerate(words[0])])
    best_word = words[0]
    for word in words:
        score = sum([rankings[idx][letter] for idx, letter in enumerate(word)])
        
        if score < best_score:
            best_score = score
            best_word = word

    return best_word
            




def main():
    args = parse_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', 
        filename='log.txt', filemode='w', level=numeric_level)

    # seeding
    if args.seed == -1 or args.seed is None:
        random.seed()
    else:
        random.seed(args.seed)

    # read in word list and preprocess
    word_list = read_word_list(args.word_list)
    word_list = filter_word_list(word_list, length=args.length)

    # get first guess
    first_guess = None
    if args.smart_first_guess:
        first_guess = smart_first_guess(word_list)
    
    # select policy
    policy = None
    if args.policy == 'random':
        policy = RandomGuessPolicy()
    elif args.policy == 'smart_random':
        policy = RandomGuessPolicy(reduce=True)
    elif args.policy == 'minimax':
        policy = MinimaxGuessPolicy(first_guess=first_guess)
    elif args.policy == 'prob_greedy':
        policy = ProbabalisticGreedyGuessPolicy(first_guess=first_guess)
    elif args.policy == 'genetic':
        policy = GeneticGuessPolicy(first_guess=first_guess, population_size=1000, max_generations=100,
                                    max_generation_size=1000)

    # create game and simulator
    game = Wordle(word_list, max_iter=args.guesses)
    sim = Simulator(game, policy)

    # run games
    if args.multiprocessing:
        nprocs = None if args.multiprocessing == -1 else args.multiprocessing
        sim.simulate_games_multiprocessing(num_games=args.num_games, print_results=True, procs=nprocs)
    else:
        sim.simulate_games(num_games=args.num_games, print_results=True)




if __name__ == '__main__':
    main()
