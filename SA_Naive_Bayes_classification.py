# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:29:30 2020
In this script the books review from [1] are used for sentiment classification
using Naive Bayes.

Only reviews with 5-star rating and 1-star rating are used in this project.

The probabilities of each word occurring are smoothed using Laplacian smoothing.
The sum of the log prior and log likelihood provide a score for classification 
into positive and negative sentiment. Reviews resulting in positive scores 
are classified as positive and reviews with negative score are classified as 
negative.

The naive bayes model provides an accuracy of approx. 82% over the test data set.

A word visualisation function is also provided based on the log of the words 
total frequency versus the log of the positive and negative probabilities ratio.
The former provides information about the words repetition rate in both 
positive and negative reviews. The latter gives information about how positive
(larger positive log prob ratio) or negative (larger negative log prob ratio) is.
 
References:
    [1] John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, 
        Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. 
        Association of Computational Linguistics (ACL), 2007
        (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

@author: Marina Torrente Rodriguez
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from SA_utils import get_reviews_dictionary
from SA_utils import get_clean_word_tokens
from SA_utils import get_vocab_freqs
from SA_utils import get_all_reviews
from sklearn.model_selection import train_test_split 

# Get positive reviews stored in a dictionary
file_name_pos = os.getcwd() + "/books/positive.review"
file_positive = open(file_name_pos, "r")
reviews_pos = get_reviews_dictionary(file_positive)
file_positive.close()   

# Get negative reviews stored in a dictionary
file_name_neg = os.getcwd() + "/books/negative.review"
file_negative = open(file_name_neg, "r")
reviews_neg = get_reviews_dictionary(file_negative)
file_negative.close()   


# Total number of positive and negative words
def count_vocabulary_total_freqs(vocab):
    """
    Input: 
        - vocab: vocabulary dictionary of keys corresponding to words and 
        values to their frequencies
    Output:
        - N: sum of frequencies of all words in the vocabulary
    """
    N = 0
    for key in vocab.keys():
        N += vocab[key]
    return N


# Define function to calculate Laplacian Smoothed Probabilities
def calculate_probabilities(freq_wi_class, N_class, V):
    """
    Input:
        - freq: frequency if appearance of a word for class positive or negative
        - N: sum of frequencies of all words in the vocabulary
        - V: length of vocabulary
    Output:
        - p_wi_class: Laplacian probability of the word for class positive or negative
    """
    p_wi_class = (freq_wi_class + 1) / (N_class + V)
    return p_wi_class

# Define function to provide words probabilities from positive and negative reviews
def get_pos_neg_probabilities(w, vocab_pos, vocab_neg, V):
    """
    Input:
        - w: word token
        - vocab_pos: vocabulary dictionary from positive reviews containing the frequency of words
        - vocab_neg: vocabulary dictionary from negative reviews containing the frequency of words
        - V: length of the entire vocabulary
    Output:
        - p_wi_1: Laplacian probability of the word for positive reviews
        - p_wi_0: Laplacian probability of the word for negative reviews
    """
    
    # Positive probability
    if (w,1) in vocab_pos:
        p_wi_1 =calculate_probabilities(vocab_pos[(w, 1)], Npos   , V)
    else:
        p_wi_1 =calculate_probabilities(0, Npos   , V)
    
    # Negative probability
    if (w,0) in vocab_neg:
        p_wi_0 =calculate_probabilities(vocab_neg[(w, 0)], Nneg   , V)
    else:
        p_wi_0 =calculate_probabilities(0, Nneg   , V)
        
    return p_wi_1, p_wi_0


# Naive Bayes predictions
def naive_bayes_classification(vocab_pos, vocab_neg, reviews, labels, V):
    """
    Input:
        - vocab_pos: vocabulary dictionary from positive reviews containing the frequency of words
        - vocab_neg: vocabulary dictionary from negative reviews containing the frequency of words
        - reviews: list of reviews body text
        - labels: reviews labels vector with 1's and 0's indicating positive and negative reviews position in the list respectively
    Output:
        - accuracy: accuracy of naive bayes method predictions
    """
    
    # Total number of words (only counted in vocabulary)
    V = len(vocab_pos) + len(vocab_neg)
    
    # log prior
    log_pior = np.log(len(reviews_pos) / len(reviews_neg))
        
    def naive_bayes_score(vocab_pos, vocab_neg, words):
        
        # Initialize log likelihood
        log_likelihood = 0

        # loop through each word
        for w in words:
            
            # get positive and negative probabilities
            p_wi_1, p_wi_0 = get_pos_neg_probabilities(w, vocab_pos, vocab_neg, V)
                                       
            # Update log_likelihood
            log_likelihood += np.log(p_wi_1/p_wi_0)
            
        # Get score = log_prior + log_likelihood
        score = log_pior + log_likelihood
        return score

    # Inference
    def naive_bayes_review_prediction(vocab_pos, vocab_neg, review):
        # get clean words in review
        words = get_clean_word_tokens(review)
        
        # naive bayes score
        score = naive_bayes_score(vocab_pos, vocab_neg, words)
        
        # Prediction
        if score <= 0:
            prediction = 0
        else:
            prediction = 1
            
        return prediction, score
    
    # Make predictions
    y_pred = []
    score = []
    for review in reviews:
        prediction_tmp, score_tmp = naive_bayes_review_prediction(
                vocab_pos, vocab_neg, review)
        y_pred.append(prediction_tmp)
        score.append(score_tmp)
    y_pred = np.array(y_pred).ravel()
        
    # Calculate accuracy
    accuracy = sum(labels.ravel() == y_pred)/len(labels)
    print('accuracy = {}'.format(accuracy))
    
    return accuracy


# Split train/test reviews
def split_train_test_reviews(reviews, test_size):
    
    # Split by indices
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            np.arange(len(reviews)), np.ones(len(reviews)), np.arange(len(reviews)),
            test_size=test_size , random_state=42)   
    
    # Create reviews list for train and test set
    rev_train = dict()
    rev_test = dict()
    for i, key in enumerate(reviews):
        if i in idx_train:
            rev_train[key] = reviews[key]
        else:
            rev_test[key] = reviews[key]
    
    return rev_train, rev_test

# Split reviews into train and test sets
rev_train_pos, rev_test_pos = split_train_test_reviews(reviews_pos, 0.33)
rev_train_neg, rev_test_neg = split_train_test_reviews(reviews_neg, 0.33)

# Generate vocabulary frequencies using the review in the train data set
vocab_pos = get_vocab_freqs(rev_train_pos, 1)
vocab_neg = get_vocab_freqs(rev_train_neg, 0)

# Calculate number of words in all positive and all negative reviews
Npos = count_vocabulary_total_freqs(vocab_pos)
Nneg = count_vocabulary_total_freqs(vocab_neg)

# Generate test reviews list
reviews_test, y_test = get_all_reviews(rev_test_pos, rev_test_neg)

# Join positive and negative dictionaries
vocab = vocab_pos
vocab.update(vocab_neg)

# Calculate length of joint vocabulary
V = len(vocab)

# Perform Naive Bayes classification
acc = naive_bayes_classification(vocab_pos, vocab_neg, reviews_test, y_test, V)

## Use probability ratio for words visualisation

# Define function to calculate probability ratios and visualise words by
# log(frequency) vs log(ratio)
def visualize_words_logfreq_vs_logratio(vocab, min_freq, ratio_filter):
    
    # Initialize lists for variables plotting
    words = list()
    ratio = list()
    freq = list()

    # Iterate through each word in the vocabulary
    for word, sentiment in vocab.keys():
        
        # get positive and negative probabilities
        p_wi_1, p_wi_0 = get_pos_neg_probabilities(word, vocab_pos, vocab_neg, len(vocab))
        
        # log ratio prob
        log_ratio = np.log(p_wi_1 / p_wi_0)
        
        # Filter words by frequency and ratio
        if vocab[(word, sentiment)] > min_freq and abs(log_ratio)>=ratio_filter:
            # If word not yet process, calculate ratio and store frequency
            if word not in words:
                words.append(word)
                ratio.append(log_ratio)
                freq.append(vocab[(word, sentiment)])
            # otherwise, add frequency value
            else:
                freq[words.index(word)] = freq[words.index(word)] + vocab[(word, sentiment)]
    
    # Transform ratio and freq lists into numpy arrays for plotting
    ratio = np.array(ratio)
    freq = np.array(freq)
    
    # Plot log ratio versus log frequency
    plt.figure(figsize=(16,7))
    plt.scatter(ratio, np.log(freq))
    for i, word in enumerate(words):
    	plt.annotate(word, xy=(ratio[i], np.log(freq[i])))
    plt.xlabel('log(Positive Prob./ Negative Prob.)')
    plt.ylabel('log(Frequency)')
    plt.ylim(0,6)
    plt.xlim(-5,5)
    plt.show()

# Call words plot by log ratio and log freq
visualize_words_logfreq_vs_logratio(vocab, 30, 0.5)
plt.title("Minimum frequency {} and minimum abs(log(ratio)) {}".format(30, 0.5))
