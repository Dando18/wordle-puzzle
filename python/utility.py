import logging


OPTIMAL_GENETIC_PARAMS = {
    'crossover_prob': 0.7, #0.31205820872312595, 
    'diversify': True, 
    'fitness_const': 7.0, 
    'invert_prob': 0.025779667955558305, 
    'max_eligible_size': 2315.0, #500.0, 
    'max_generations': 50, 
    'mutate_prob': 0.09457252541449512, 
    'permute_prob': 0.06168892887298874, 
    'population_size': 2315.0, #1000.0, 
    'tournament_size': 100.0 #90.0
    }


def read_word_list(fname):
    with open(fname, 'r') as fp:
        if fname.endswith('.json'):
            from json import load
            return list(load(fp).keys())
        else:
            return list(map(lambda x: x.strip(), fp.readlines()))


def filter_word_list(words, length=5):
    return list( filter(lambda x: len(x) == length, words) )



def remove_impossible_words_(last_state, word_list):
    ''' Several guess policies use this, so just a helper for code re-use.
    '''
    guess, result = last_state

    # correct letters, keep these
    correct_letters = [(idx, l) for idx, (l, c) in enumerate(zip(guess, result)) if c == 'CORRECT']

    # remove words with these letters in them
    removed_letters = [l for l, c in zip(guess, result) if c == 'INCORRECT']
    
    # these letters can't be in these positions
    misplaced_letters = [(idx, l) for idx, (l, c) in enumerate(zip(guess, result)) if c == 'MISPLACED']

    # filter test func
    def should_keep_word(word):
        # remove words with incorrect letters
        if any(letter in word for letter in removed_letters):
            return False

        # remove words if they have misplaced letters in the wrong spot
        if any( (word[idx] == letter) for idx, letter in misplaced_letters ):
            return False
        
        # remove words that don't match correct letters
        if any( (word[idx] != letter) for idx, letter in correct_letters ):
            return False

        return True

    new_possible_guesses = list( filter(should_keep_word, word_list) )

    total_removed = len(word_list) - len(new_possible_guesses)
    logging.debug('Removed {} words from list. {} remaining.'.format(total_removed, len(new_possible_guesses)))
    return new_possible_guesses


def hits(a, b):
    matches, misplaced = 0, 0
    for idx, a_char in enumerate(a):
        if a_char == b[idx]:
            matches += 1
        elif a_char in b:
            misplaced += 1
    return matches, misplaced
    

# this version is much slower
def hits_slow(a, b):
   total_matches = sum((Counter(a) & Counter(b)).values())
   same_pos_matches = sum(x == y for x, y in zip(a, b))
   misplaced_matches = total_matches - same_pos_matches
   return same_pos_matches, misplaced_matches
