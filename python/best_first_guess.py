from collections import Counter

from utility import read_word_list, filter_word_list, hits


def list_overlap(list1, list2):
    return sum([1 for x in list1 if x in list2])


def best_guess(words):
    # find the maximum average number of matches
    #print(Counter(hits('cigar', w) for w in words))
    return max(words, key=lambda x: sum(hits(x, w)[0] for w in words)/len(words))


def main():
    word_list_1 = read_word_list('../wordle-word-list-solutions.txt')
    word_list_2 = read_word_list('../wordle-word-list-non-solutions.txt')
    alpha_list = read_word_list('../english-words/words_dictionary.json')
    alpha_list = filter_word_list(alpha_list)

    print('list 1 length: {}'.format(len(word_list_1)))
    print('list 2 length: {}'.format(len(word_list_2)))
    print('total: {}'.format(len(word_list_1) + len(word_list_2)))
    
    print('alpha list length: {}\n'.format(len(alpha_list)))


    list_1_overlap_w_list_2 = list_overlap(word_list_1, word_list_2)
    print('List 1 shares {} words with list 2.'.format(list_1_overlap_w_list_2))

    list_1_overlap_w_alpha = list_overlap(word_list_1, alpha_list)
    print('List 1 shares {} words with alpha list.'.format(list_1_overlap_w_alpha))

    list_2_overlap_w_alpha = list_overlap(word_list_2, alpha_list)
    print('List 2 shares {} words with alpha list.'.format(list_2_overlap_w_alpha))
    print('total: {}\n'.format(list_1_overlap_w_alpha + list_2_overlap_w_alpha))


    best_word_list_1 = best_guess(word_list_1)
    print('List 1 best word: {}'.format(best_word_list_1))

    best_word_list_2 = best_guess(word_list_2)
    print('List 2 best word: {}'.format(best_word_list_2))

    best_word_alpha_list = best_guess(alpha_list)
    print('Alpha l best word: {}'.format(best_word_alpha_list))



if __name__ == '__main__':
    main()