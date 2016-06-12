"""
Module to load the Stanford Sentiment Treebank (SSTb) dataset.

Usage:  glove =	GloveEncoder(glove_path,WORD_DIM,PHRASE_LEN)
        sstb = SSTb(MAX_NEG, MIN_POS, glove, sstb_path)
        train,val,test = sstb.load_ds(COUNT, TRAIN_FRAC,VAL_FRAC)

MAX_NEG, MIN_POS set the thresholds for sentiment classes.
For example, MAX_NEG=0.4 will treat sentiment polarity values in [0,0.4) as NEGATIVE
             MIN_POS=0.6 will treat sentiment polarity values [0.6,1) as POSITIVE
             All other polarities (in [0.4,0.6)) will be considered NEUTRAL

TRAIN_FRAC, VAL_FRAC should be fractions indicating portions of the dataset used for the training and validation sets
For example, for TRAIN_FRAC=0.7 and VAL_FRAC=0.1:
                Training set- records 0 to 0.7*COUNT (length=0.7*COUNT)
                Validation set- records 0.7*COUNT to 0.8*COUNT (length=0.1*COUNT)
                Test set- records 0.8*COUNT to COUNT

train, val and test are each 2-tuples of the form (X,Y)

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""
import numpy as np
import random
from glove_encoder import GloveEncoder

class SSTb():

    def __init__(self, max_neg, min_pos, glove, path):
        #Positions of the output 3-value vector indicating sentiment class
        self.NEG = 0
        self.NET = 1
        self.POS = 2
        #Thresholds for sentiment classes
        self.max_neg = max_neg
        self.min_pos = min_pos
        self.glove = glove
        #Paths to phrases and sentiment labels in the SSTb folder
        self.phrase_path = path+'\\dictionary.txt'
        self.sent_path = path+'\\sentiment_labels.txt'

    def clean(self,line):
        return ''.join([c for c in line if ord(c) < 128])

    #Method to load the dataset, with the given number of records and size of training, validation sets
    def load_ds(self, count, tf, vf):
        X = []
        Y = []
        p_id = {}
        p_sent = {}
        #Load maps: p_id- phrase ID->phrase, p_sent: phrase ID->sentiment polarity
        for line in open(self.phrase_path).readlines()[:count]:
            line = self.clean(line).strip()
            p = line.split('|')
            p_id[int(p[1])] = p[0]
        for line in open(self.sent_path).readlines()[1:]:
            line = self.clean(line).strip()
            p = line.split('|')
            p_sent[int(p[0])] = float(p[1])
        #Map phrases to sentiment polarities
        for id in p_id.keys():
            em = self.glove.encode_phrase(p_id[id])
            if not em is None:
                X.append(em)
            else:
                del p_id[id]
                del p_sent[id]
                continue
            sent = self.NEG
            if id in p_sent:
                if p_sent[id] < self.max_neg:
                    sent = self.NEG
                elif p_sent[id] >= self.min_pos:
                    sent = self.POS
                else:
                    sent = self.NET
            out = np.array([0,0,0])
            out[sent] = 1
            Y.append(out)
        ds = zip(X,Y)
        random.shuffle(ds)
        X,Y = zip(*ds)
        X = np.array(X)
        Y = np.array(Y)

        #Split dataset to training, validation and test sets
	trainX = X[:int(X.shape[0]*tf),:,:]
	trainY = Y[:int(Y.shape[0]*tf),:]
	valX = X[int(X.shape[0]*tf):int(X.shape[0]*(tf+vf)),:,:]
	valY = Y[int(Y.shape[0]*tf):int(Y.shape[0]*(tf+vf)),:]
	testX = X[int(X.shape[0]*(tf+vf)):,:,:]
	testY = Y[int(Y.shape[0]*(tf+vf)):,:]
	print 'Train, Val, Test'
	print trainX.shape,',',trainY.shape,'--',valX.shape,',',valY.shape,'--',testX.shape,',',testY.shape
        return ((trainX,trainY),(valX,valY),(testX,testY))
