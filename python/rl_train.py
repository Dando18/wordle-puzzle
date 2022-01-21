from itertools import repeat
from random import randrange

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

from wordle import Wordle
from wordle_simulator import Simulator
from wordle_guess_policy import RLGuessPolicy


def read_word_list(fname):
    with open(fname, 'r') as fp:
        if fname.endswith('.json'):
            from json import load
            return list(load(fp).keys())
        else:
            return list(fp.readlines())


def filter_word_list(words, length=5):
    return list( filter(lambda x: len(x) == length, words) )


class WordleEnv(gym.Env):

    def __init__(self, game):
        super(WordleEnv, self).__init__()
        self.game_ = game

        self.last_score_ = 0
        self.word_length_ = self.game_.word_length()
        self.num_actions_ = len(self.game_.word_list())

        self.action_space = spaces.Discrete(self.num_actions_)
        self.observation_space = spaces.MultiDiscrete(list(repeat(3, self.word_length_)))


    def get_random_index(self):
        return randrange(self.num_actions_)

    
    def reset(self):
        self.last_score_ = 0
        self.game_.reset(update_word=True)
        selected_word_idx = self.get_random_index()
        return np.array(list(repeat(2, self.word_length_))).astype(np.int32)

    
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
        observation = self.result_to_observation(result)

        return observation, reward, done, info




def main():
    # read in word list and preprocess
    word_list = read_word_list('../english-words/words_dictionary.json')
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