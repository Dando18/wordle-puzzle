''' Wordle simulator and/or AI
author: Daniel Nichols
date:  January 2021
'''
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_arg('--word-list', type=str, default='word-list.txt')
    return parser.parse_args()


def read_word_list(words):
    pass

def main():
    args = parse_args()


if __name__ == '__main__':
    main()
