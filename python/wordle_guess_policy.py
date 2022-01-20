from abc import ABC, abstractmethod
from collections import Counter
from itertools import product
import logging
from random import choice, sample, randrange, random
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
        ''' Returns the maximum # of possible remaining words left if you were to guess word.
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


class GeneticGuessPolicy(GuessPolicy):

    def __init__(self, first_guess=None, population_size=150, max_generations=100, max_generation_size=250):
        super().__init__('genetic')

        self.first_guess_ = first_guess
        self.population_size_ = population_size
        self.max_generations_ = max_generations
        self.max_generation_size_ = max_generation_size
        self.guess_count_ = 0


    def remove_impossible_words(self, last_state):
        self.possible_guesses_ = remove_impossible_words_(last_state, self.possible_guesses_)


    def hits(self, a, b):
        total_matches = sum((Counter(a) & Counter(b)).values())
        same_pos_matches = sum(x == y for x, y in zip(a, b))
        misplaced_matches = total_matches - same_pos_matches
        return same_pos_matches, misplaced_matches


    def fitness(self, word, game_state, const=2):
        sum = 0
        for guess, result in game_state:
            match, misplaced = result.count('CORRECT'), result.count('MISPLACED')
            pos_match, pos_misplaced = self.hits(guess, word)

            sum += abs(pos_match - match) + abs(pos_misplaced - misplaced)

        return -(sum + const*len(word)*(self.guess_count_-1))


    def selection(self, fitnesses, population, k=20, keep_size=True):
        def random_tournament():
            selected_idx = randrange(len(population))

            for idx in sample(range(len(population)), k=k):
                if fitnesses[idx] > fitnesses[selected_idx]:
                    selected_idx = idx
            return population[selected_idx]

        return [random_tournament() for _ in range(len(population))]


    def crossover(self, parent1, parent2, prob=0.5):
        child1, child2 = parent1, parent2

        if random() < prob:
            split = randrange(1, len(parent1)-1)
            child1 = parent1[:split] + parent2[split:]
            child2 = parent2[:split] + parent1[split:]

        return [child1, child2]

    
    def mutate(self, word, prob=0.03):
        if random() < prob:
            word = list(word)
            word[randrange(len(word))] = choice(ascii_lowercase)
            word = ''.join(word)
        return word


    def permute(self, word, prob=0.03):
        if random() < prob:
            idx1, idx2 = sample(range(len(word)), 2)
            word = list(word)
            word[idx1], word[idx2] = word[idx2], word[idx1]
            word = ''.join(word)
        return word


    def invert(self, word, prob=0.02):
        if random() < prob:
            idx1, idx2 = sample(range(len(word)), 2)
            idx1, idx2 = min(idx1,idx2), max(idx1,idx2)
            word = list(word)
            word[idx1:idx2] = reversed(word[idx1:idx2])
            word = ''.join(word)
        return word


    def next_guess(self, game_state, game):
        # handle first_guess
        if self.guess_count_ == 0:
            self.possible_guesses_ = game.word_list().copy()
            self.guess_count_ += 1

            return self.first_guess_ if self.first_guess_ else choice(game.word_list())

        self.guess_count_ += 1

        # remove words which cannot be used
        #self.remove_impossible_words(game_state[-1])

        # create population
        population = sample(self.possible_guesses_, self.population_size_)
        eligible_words = set()
        generation = 0

        # do genetic iterations
        while generation < self.max_generations_ and len(eligible_words) < self.max_generation_size_:
            # selection
            fitnesses = [self.fitness(p, game_state) for p in population]
            selected = self.selection(fitnesses, population, k=40)

            # new generation
            new_pop = []
            for p1, p2 in zip(selected[0::2], selected[1::2]):
                for c in self.crossover(p1, p2, prob=0.5):
                    c = self.mutate(c, prob=0.03)
                    c = self.permute(c, prob=0.03)
                    c = self.invert(c, prob=0.02)

                    if c not in eligible_words and c in self.possible_guesses_:
                        new_pop.append(c)
                    else:
                        new_pop.append( choice(self.possible_guesses_) )

            population = new_pop
            eligible_words.update(population)

            generation += 1

        # choose word in eligible_words with maximum
        best_word = max( eligible_words, key=lambda x: self.fitness(x, game_state) )
        return best_word


    def reset(self):
        self.guess_count_ = 0
