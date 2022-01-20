import logging

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
