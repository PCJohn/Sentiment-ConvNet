"""
Module to encode text with GloVe vectors.
The encoder maps every word to its GloVe word embedding and stacks these vectors vertically to form a matrix.

Usage:  glove = GloveEncoder('glove\\glove.6B.50d.txt', word_dim=50, max_len=20)
        text = 'This is a test sentence.'
        enc_matrix = glove.encode_phrase(text)

word_dim specifies the length of each word embedding. This can be 50, 100, 200 or 300 (the available lengths for GloVe vectors).
max_len is the number of words in the phrase. To ensure that all inputs have input matrices with consistent dimensions, this considers a fixed number of words of the input text.
E.g: If max_len=5, the encoder considers the first 5 words of the text. If the text has less  than 5 words, the encoded matrix is padded with 0-vectors to make it have 5 rows.

enc_matrix will be a numpy array with shape (max_len, word_dim)

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""

import numpy as np
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

class GloveEncoder():
    def __init__(self, glove_path, word_dim, max_len):
        self.word_dim = word_dim
        self.max_len = max_len
        self.lm = WordNetLemmatizer()
        glove_path = glove_path+'\\glove.6B.'+str(word_dim)+'d.txt.'
        self.glove_map = dict(map(self.split,open(glove_path).readlines()))

    def split(self,line):
            p = line.split(' ')
            return (p[0].lower(),np.array([float(f) for f in p[1:]]))

    def encode_phrase(self,phrase):
            em = np.array([])
            for w in word_tokenize(phrase)[:self.max_len]:
                    w = self.lm.lemmatize(w.lower())
                    if w in self.glove_map:
                            if em.size == 0:
                                    em = np.reshape(self.glove_map[w],(1,self.word_dim))
                            else:
                                    em = np.vstack((em,(np.reshape(self.glove_map[w],(1,self.word_dim)))))
            if em.size == 0:
                    return None
            while em.shape[0] < self.max_len:
                    em = np.vstack((em,np.zeros(shape=(1,self.word_dim))))
            return em
