# Sentiment-ConvNet
Convolutional neural networks for sentiment classification of short texts.

This includes a set of scripts for sentiment classifcation with a model loosely based on "Deep Convolutional Neural Networks for
Sentiment Analysis of Short Texts" by dos Santos et al. (http://www.aclweb.org/anthology/C14-1008)

Training set:
11976 phrases from the Stanford Sentiment Treebank dataset or "SSTb" (zip included here). Every phrase is encoded into a representative matrix with dense 50 dimensional GloVe embeddings (zip included). The module sets a fixed limit of  15 words per sentence to ensure that the embedding matrix of sentences have a uniform dimension of 15X50.

Pre-trained models:
There are two pre-trained models included. Each has been tested on 3992 phrases (distinct from the training set) of SSTb. sstb_2class.pkl is a binary classification model (positive-negative). The final test accuracy achieved is 90.	003%
sstb_3class.pkl is a 3 class model (positive, neutral and negative) that is trained with the same architecture and parameters as the previous model. The sentiment polarity is split into casses as: [0,0.33) - negative, [0.33,67) - neutral, [0.67,1] - positive. The final accuracy is 66.64% (low because of no tuning specifically for 3 way classification).

Usage:
Extract SSTb.zip and glove.zip to folders SSTb and glove. Run python sentiment.py
to run the default training session save the model to "sentiment_model.pkl".
See documentation within the modules for detailed usage.

Requirements:

    1) theano, lasagne
    2) scikit-neuralnetwork
    3) Python 2.7 (64 bit)

Here's the output of interactive_test() in sentiment.py:

![Alt text](conv_sent_screenshot.JPG?raw=true)
