''' Simulate a Mastermind/Wordle like game and record stats.
    author: Daniel Nichols
    date: January 2022
'''
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

    
    def simulate_games(self, num_games=1, show_progress=True, print_results=False):
        all_stats = []
        start = time.time()
        bar = alive_it(range(num_games), enrich_print=False, disable=(not show_progress))
        guess_sum = 0
        for idx in bar:
            logging.info('Simulating iteration {}.'.format(idx))

            stats = self.simulate_game()
            all_stats.append(list(stats))
            guess_sum += stats[1]
            bar.text('Avg. # Guesses: {:.3f}'.format(guess_sum / (idx+1)))

            self.game_.reset(update_word=True)
            self.policy_.reset()

        end = time.time()
        all_stats = np.array(all_stats)
        if print_results:
            self.print_simulation_results_(all_stats, num_games, (end-start))
        
        return float(np.sum(all_stats, axis=0)[1]) / num_games


    def simulate_games_multiprocessing(self, num_games=1, show_progress=True, print_results=False, procs=None):
        from itertools import repeat
        import multiprocessing as mp

        if procs is None:
            from os import cpu_count
            procs = cpu_count()
        logging.info('Using {} cpus to simulate {} games.'.format(procs, num_games))

        mean_guesses = None
        with mp.Pool(procs) as pool, alive_bar(num_games, enrich_print=False, disable=(not show_progress)) as bar:
            start = time.time()
            
            all_stats = pool.imap_unordered(run_simulation_step, [(self, idx) for idx in range(num_games)], 
                chunksize=num_games//(procs**2)+1)

            tmp = []
            guess_sum = 0
            for s in all_stats:
                tmp.append(s)
                guess_sum += s[1]
                bar()
                bar.text('Avg. # Guesses: {:.3f}'.format(guess_sum / len(tmp)))
            all_stats = np.array(tmp)

            end = time.time()
            mean_guesses = float(np.sum(all_stats, axis=0)[1]) / num_games
            if print_results:
                self.print_simulation_results_(all_stats, num_games, (end-start))
            
        return mean_guesses

    
    def print_simulation_results_(self, all_stats, num_games, duration):
        cum_stats = np.sum(all_stats, axis=0)
        perc_le_6 = (all_stats[:,1] <= 6).sum() / num_games * 100.0

        print('# Games: {}'.format(num_games))
        print('# Wins:  {}'.format(cum_stats[0]))
        print('% <= 6 Guesses:    {:.3f}'.format(perc_le_6))
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
        print('policy,num_games,num_wins,perc_le_6,avg_time,max_guesses,min_guesses,avg_guesses,median_guesses,std_guesses,' + 
            'perc_correct_letters,perc_misplaced_letters,perc_incorrect_letters')
        print(','.join(map(str, 
                [self.policy_.name_, num_games, cum_stats[0], perc_le_6, duration/num_games, all_stats[:,1].max(), 
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

            if result is None:
                break

            num_guesses += 1
            num_letters_guessed += len(guess)
            num_correct_letters_guessed += result.count('CORRECT')
            num_misplaced_letters_guessed += result.count('MISPLACED')
            num_incorrect_letters_guessed += result.count('INCORRECT')

            logging.debug('guess {}:  {} -> {}'.format(num_guesses, guess, result))

            if result and result.count('CORRECT') == len(result):
                keep_guessing = False
                correct_answer = True

        return correct_answer, num_guesses, num_letters_guessed, num_correct_letters_guessed, \
            num_misplaced_letters_guessed, num_incorrect_letters_guessed

