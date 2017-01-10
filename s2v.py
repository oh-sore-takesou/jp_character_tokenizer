'''
Japanese to char-level one-hot vector encoder
'''

import pykakasi.kakasi
import numpy as np
import MeCab

from token_dictionary import alphabet, katakana, katakana_small

class J2V:

    '''
    dictionary_type: (str) ./token_dictionary
    max_len: (int)
    '''
    def __init__(self, dictionary_type='alphabet', max_len=1000, length=1024):
        if dictionary_type == 'alphabet':
            self.dictionary = alphabet.alphabet_dict
        elif dictionary_type == 'katakana':
            self.dictionary = katakana.katakana_dict
        elif dictionary_type == 'katakana_small':
            self.dictionary = katakana_small.katakana_small_dict
        self.dictionary_type = dictionary_type
        self.max_len = max_len
        self.length = length
        kakasi = pykakasi.kakasi()
        kakasi.setMode('H', 'a')
        kakasi.setMode('K', 'a')
        kakasi.setMode('J', 'a')
        self.kakasi_conv = kakasi.getConverter()

    '''
    input: ['日本語の文章', ...]
    output: [[[0,0,1,...], [0,1,...], ...]]
    '''
    def encode_sentences(self, sentences):
        if type(sentences) != type(list()):
            sentences = [sentences]

        return [ self.__encode_sentence(sentence) for sentence in sentences ]

    '''
    input: '日本語の文章'
    output: [[0,0,1,...], [0,1,0,...]]
    '''
    def __encode_sentence(self, sentence):

        tokenized_sentence = [ char for char in self.__tokenize(sentence) ]
        while len(tokenized_sentence) < self.length:
            tokenized_sentence.append([0]*len(self.dictionary))
        
        return tokenized_sentence

    '''
    input: '日本語の文章'
    output: ['ﾆ', 'ﾎ', 'ﾝ', ...] or ['n', 'i', 'h', 'o', ...]
    '''
    def __tokenize(self, sentence):

        if self.dictionary_type == 'alphabet':
            return [ self.__encode_char(char) for char in self.kakasi_conv.do(sentence) ][:self.max_len]
        elif self.dictionary_type in ['katakana', 'katakana_small']:
            return [ self.__encode_char(char) for char in sentence ][:self.max_len]

    '''
    input: '日'
    output: '[0,1,..]'  OR [zeros] if doesn't exist in dictionary
    '''
    def __encode_char(self, char):

        char_vec = [0] * len(self.dictionary)
        try:
            token_id = self.dictionary[char.lower()]
            char_vec[token_id-1] = 1
        except KeyError:
            return char_vec

        return  char_vec


if __name__ == '__main__':
    j2v = J2V('katakana_small', max_len=1000, length=1024)
    r = j2v.encode_sentences(['日本', 'America'])
    print(len(r[0]))
    print(len(r[0][0]))
