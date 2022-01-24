''' Train an RL model to play Mastermind/Wordle like games.
    author: Daniel Nichols
    date: January 2022
'''
from itertools import repeat
from random import randrange

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

from wordle import Wordle
from simulator import Simulator
from wordle_guess_policy import RLGuessPolicy
from utility import read_word_list, filter_word_list


class WordleEnv(gym.Env):

    def __init__(self, game):
        super(WordleEnv, self).__init__()
        self.game_ = game

        self.last_score_ = 0
        self.word_length_ = self.game_.word_length()
        self.num_actions_ = len(self.game_.word_list())
        self.encoded_game_state_ = list(repeat(26, self.word_length_))  # all unknown

        self.action_space = spaces.Discrete(self.num_actions_)
        #self.observation_space = spaces.MultiDiscrete(list(repeat(3, self.word_length_)))
        control_lengths = list(repeat(26 + 1, self.word_length_))
        self.observation_space = spaces.MultiDiscrete(control_lengths)


    def get_random_index(self):
        return randrange(self.num_actions_)

    
    def updated_encoded_game_state(self, last_state):
        guess, result = last_state
        for idx, (letter, res) in enumerate(zip(guess, result)):
            dig = ord(letter) - 97
            if res == 'CORRECT':
                self.encoded_game_state_[idx] = dig

    
    def reset(self):
        self.last_score_ = 0
        self.encoded_game_state_ = list(repeat(26, self.word_length_))  # all unknown
        self.game_.reset(update_word=True)
        selected_word_idx = self.get_random_index()
        return np.array(self.encoded_game_state_).astype(np.int32)

    
    def result_to_observation(self, result):
        vals = {'CORRECT': 0, 'MISPLACED': 1, 'INCORRECT': 2}
        return np.array([vals[x] for x in result]).astype(np.int32)


    def step(self, action):
        if action < 0 or action >= self.num_actions_:
            raise ValueError('Invalid action.')

        # get the word guessed
        selected_word = self.game_.word_list()[action]

        # get output
        result = self.game_.guess(selected_word)
        matches, misplaced = result.count('CORRECT'), result.count('MISPLACED')

        # are we done yet?
        done = bool(matches == self.word_length_)

        # reward the good boi
        score = matches + misplaced
        reward = 1 if score > self.last_score_ else 0
        self.last_score_ = score

        # more info
        info = {}

        # observation
        #observation = self.result_to_observation(result)
        self.updated_encoded_game_state((selected_word, result))
        observation = np.array(self.encoded_game_state_).astype(np.int32)

        return observation, reward, done, info




def main():
    # read in word list and preprocess
    #word_list = read_word_list('../english-words/words_dictionary.json')
    word_list = read_word_list('../wordle-word-list-solutions.txt')
    word_list = filter_word_list(word_list, length=5)

    # setup game and environment
    game = Wordle(word_list)
    env = WordleEnv(game)

    # do quick check on environment
    check_env(env, warn=True)

    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000, log_interval=1)

    game = Wordle(word_list, max_iter=100)
    policy = RLGuessPolicy(model)
    sim = Simulator(game, policy)
    sim.simulate_games(num_games=100, print_results=True)


if __name__ == '__main__':
    main()