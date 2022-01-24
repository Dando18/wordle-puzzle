''' Game agent for the game of Wordle.
    author: Daniel Nichols
    date: January 2022
'''
import logging
from random import choice

class Wordle:

    def __init__(self, words, max_iter=None, word=None):
        self.words_ = words
        self.max_iter_ = max_iter
        self.cur_iter_ = 0

        if word is None:
            self.word_ = choice(self.words_)
            logging.debug('Wordle selected word: {}'.format(self.word_))
        else:
            self.word_ = word
    

    def guess(self, word):
        word = word.lower()

        if word not in self.words_:
            raise ValueError('invalid word')

        if self.max_iter_ is not None and self.cur_iter_ >= self.max_iter_:
            return None

        self.cur_iter_ += 1

        ans = []
        for idx, letter in enumerate(word):

            if letter == self.word_[idx]:
                ans.append('CORRECT')
            elif letter in self.word_:
                ans.append('MISPLACED')
            else:
                ans.append('INCORRECT')
        
        return ans

    
    def reset(self, update_word=False):
        self.cur_iter_ = 0

        if update_word:
            self.word_ = choice(self.words_)
            logging.debug('Wordle selected word: {}'.format(self.word_))

    
    def word_list(self):
        return self.words_

    def word_length(self):
        return len(self.words_[0])

