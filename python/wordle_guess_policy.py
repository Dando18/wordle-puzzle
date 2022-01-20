from abc import ABC, abstractmethod
from collections import Counter
from itertools import product
import logging
from random import choice
from string import ascii_lowercase

from utility import remove_impossible_words_


class GuessPolicy(ABC):

    def __init__(self, name):
        super().__init__()
        self.name_ = name

    @abstractmethod
    def next_guess(self, game_state, game):
        pass

    @abstractmethod
    def reset(self):
        pass


class RandomGuessPolicy(GuessPolicy):

    def __init__(self, reduce=False):
        super().__init__('smart_random' if reduce else 'random')

        self.reduce_ = reduce
        self.guess_count_ = 0
        self.possible_guesses_ = []


    def next_guess(self, game_state, game):
        if self.guess_count_ == 0:
            self.possible_guesses_ = game.word_list().copy()
        self.guess_count_ += 1

        if self.reduce_ and self.guess_count_ != 1:
            self.possible_guesses_ = remove_impossible_words_(game_state[-1], self.possible_guesses_)

        return choice(self.possible_guesses_)


    def reset(self):
        self.guess_count_ = 0
        self.possible_guesses_ = []


class MinimaxGuessPolicy(GuessPolicy):

    def __init__(self, first_guess=None):
        super().__init__('minimax')
        self.first_guess_ = first_guess
        self.guess_count_ = 0


    def remove_impossible_words(self, last_state):
        self.possible_guesses_ = remove_impossible_words_(last_state, self.possible_guesses_)


    def score(self, word):
        ''' Returns the maximum # of remaining possibilities.
        '''
        def hits(a, b):
            total_matches = sum((Counter(a) & Counter(b)).values())
            same_pos_matches = sum(x == y for x, y in zip(a, b))
            misplaced_matches = total_matches - same_pos_matches
            return same_pos_matches, misplaced_matches

        # max number of hits 
        return max( Counter(hits(word, x) for x in self.possible_guesses_).values() )


    def next_guess(self, game_state, game):
        # handle first_guess
        if self.guess_count_ == 0:
            self.possible_guesses_ = game.word_list().copy()
            self.guess_count_ += 1

            return self.first_guess_ if self.first_guess_ else choice(game.word_list())
        
        self.guess_count_ += 1

        # remove words which cannot be possible
        self.remove_impossible_words(game_state[-1])

        # do minimax on remaining words
        best_score_word = min(self.possible_guesses_, key=self.score)

        return best_score_word


    def reset(self):
        self.guess_count_ = 0
        self.possible_guesses_ = []


class ProbabalisticGreedyGuessPolicy(GuessPolicy):

    def __init__(self, first_guess=None):
        super().__init__('probabilistic_greedy')

        self.first_guess_ = first_guess
        self.guess_count_ = 0


    def remove_impossible_words(self, last_state):
        self.possible_guesses_ = remove_impossible_words_(last_state, self.possible_guesses_)


    def likelihood(self, letters):

        num_matches = 0
        for w in self.possible_guesses_:
            is_match = all( w[idx] == l for idx, l in letters )
            if is_match:
                num_matches += 1

        return float(num_matches) / len(self.possible_guesses_)


    def substitute(self, letters, word):
        word = list(word)
        for idx, l in letters:
            word[idx] = str(l)
        return ''.join(word)


    def next_guess(self, game_state, game):
        # handle first_guess
        if self.guess_count_ == 0:
            self.possible_guesses_ = game.word_list().copy()
            self.guess_count_ += 1
            self.remaining_letters_ = list(ascii_lowercase)

            return self.first_guess_ if self.first_guess_ else choice(game.word_list())
        
        self.guess_count_ += 1
        last_state = game_state[-1]

        # remove words which cannot be possible
        self.remove_impossible_words(last_state)

        # update available letters
        valid_indices = [idx for idx, c in enumerate(last_state[1]) if c != 'CORRECT']
        incorrect_letters = [l for l, c in zip(*last_state) if c == 'INCORRECT']
        for l in incorrect_letters:
            if l not in self.remaining_letters_:
                self.remaining_letters_.remove(l)

        # choose max likelihood
        likelihoods = []
        for idx in valid_indices:
            tmp_likelihood = [(l, self.likelihood([(idx, l)])) for l in self.remaining_letters_]
            tmp_likelihood.sort(key=lambda x: x[1], reverse=True)
            likelihoods.append( tmp_likelihood )


        best_word = None
        for possibility in product(*likelihoods):
            possibility = [(idx, letter) for idx, (letter, lik) in zip(valid_indices, possibility)]
            best_word = self.substitute(list(possibility), last_state[0])
            
            if best_word in self.possible_guesses_:
                break
        
        return best_word


    def reset(self):
        self.guess_count_ = 0
        self.possible_guesses_ = []
