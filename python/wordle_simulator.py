
from collections import Counter
import logging
from random import choice
import time

from alive_progress import alive_it, alive_bar
import numpy as np

def run_simulation_step(args):
    simulator, idx = args
    logging.info('Simulating iteration {}.'.format(idx))

    simulator.game_.reset(update_word=True)
    simulator.policy_.reset()
    return simulator.simulate_game()


class Simulator:

    def __init__(self, game, policy):
        self.game_ = game
        self.policy_ = policy

    
    def simulate_games(self, num_games=1, print_results=False):
        all_stats = []
        start = time.time()
        for idx in alive_it(range(num_games), enrich_print=False):
            logging.info('Simulating iteration {}.'.format(idx))

            stats = self.simulate_game()
            all_stats.append(list(stats))

            self.game_.reset(update_word=True)
            self.policy_.reset()

        end = time.time()
        all_stats = np.array(all_stats)
        if print_results:
            self.print_simulation_results_(all_stats, num_games, (end-start))


    def simulate_games_multiprocessing(self, num_games=1, print_results=False, procs=None):
        from itertools import repeat
        import multiprocessing as mp

        if procs is None:
            from os import cpu_count
            procs = cpu_count()

        with mp.Pool(procs) as pool, alive_bar(num_games, enrich_print=False) as bar:
            start = time.time()
            
            all_stats = pool.imap_unordered(run_simulation_step, [(self, idx) for idx in range(num_games)], 
                chunksize=num_games//(procs**2)+1)

            tmp = []
            for s in all_stats:
                tmp.append(s)
                bar()
            all_stats = np.array(tmp)

            end = time.time()
            if print_results:
                self.print_simulation_results_(all_stats, num_games, (end-start))

    
    def print_simulation_results_(self, all_stats, num_games, duration):
        cum_stats = np.sum(all_stats, axis=0)

        print('# Games: {}'.format(num_games))
        print('# Wins:  {}'.format(cum_stats[0]))
        print('Avg Time per Game: {:.3f} sec'.format(duration/num_games))
        print('Max # Guesses:     {:}'.format(all_stats[:,1].max()))
        print('Min # Guesses:     {:}'.format(all_stats[:,1].min()))
        print('Avg. # Guesses:    {:}'.format(cum_stats[1] / num_games))
        print('Median # Guesses:  {:}'.format(np.median(all_stats[:,1])))
        print('Std. Dev. Guesses: {:.3f}'.format(all_stats.std(axis=0)[1]))
        print('% Correct Letters:   {:.3f}'.format(float(cum_stats[3]) / cum_stats[2] * 100))
        print('% Misplaced Letters: {:.3f}'.format(float(cum_stats[4]) / cum_stats[2] * 100))
        print('% Incorrect Letters: {:.3f}'.format(float(cum_stats[5]) / cum_stats[2] * 100))

        print()
        print('policy,num_games,num_wins,avg_time,max_guesses,min_guesses,avg_guesses,median_guesses,std_guesses,' + 
            'perc_correct_letters,perc_misplaced_letters,perc_incorrect_letters')
        print(','.join(map(str, 
                [self.policy_.name_, num_games, cum_stats[0], duration/num_games, all_stats[:,1].max(), 
                all_stats[:,1].min(), cum_stats[1] / num_games, np.median(all_stats[:,1]), all_stats.std(axis=0)[1], 
                float(cum_stats[3]) / cum_stats[2] * 100, float(cum_stats[4]) / cum_stats[2] * 100, 
                float(cum_stats[5]) / cum_stats[2] * 100]))
            )
        print()


    def simulate_game(self):
        game_state = []
        keep_guessing = True

        num_guesses = 0
        correct_answer = False
        num_letters_guessed = 0
        num_correct_letters_guessed = 0
        num_misplaced_letters_guessed = 0
        num_incorrect_letters_guessed = 0

        while keep_guessing:

            guess = self.policy_.next_guess(game_state, self.game_)

            result = self.game_.guess(guess)
            game_state.append((guess, result))

            num_guesses += 1
            num_letters_guessed += len(guess)
            num_correct_letters_guessed += result.count('CORRECT')
            num_misplaced_letters_guessed += result.count('MISPLACED')
            num_incorrect_letters_guessed += result.count('INCORRECT')

            logging.debug('guess {}:  {} -> {}'.format(num_guesses, guess, result))

            if result is None:
                keep_guessing = False

            if result and result.count('CORRECT') == len(result):
                keep_guessing = False
                correct_answer = True

        return correct_answer, num_guesses, num_letters_guessed, num_correct_letters_guessed, \
            num_misplaced_letters_guessed, num_incorrect_letters_guessed
