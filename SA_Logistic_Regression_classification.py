# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:24:13 2020

In this script the books review from [1] are used for sentiment classification
using Logistic Regression.

Only reviews with 5-star rating and 1-star rating are used in this project.

The logistic regression is implemented from scratch.

A visualisation of the word features is provided, as well as the assessment 
of the model on a test data set (resulting in 83.6% accuracy) and example of 
misclassified reviews to help understanding the limitations of the model.

@author: Marina Torrente Rodriguez

References:
    [1] John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, 
        Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. 
        Association of Computational Linguistics (ACL), 2007
        (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from SA_utils import get_reviews_dictionary
from SA_utils import get_clean_word_tokens
from SA_utils import get_vocab_freqs
from SA_utils import get_all_reviews

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


# Get word frequencies
vocab_pos = get_vocab_freqs(reviews_pos, 1)
vocab_neg = get_vocab_freqs(reviews_neg, 0)

# Get all reviews body text in a list and corresponding positive/negative labels
all_reviews, y = get_all_reviews(reviews_pos, reviews_neg)

# Define function to generate the vocabulary features' matrix 
def get_reviews_features(vocab_pos, vocab_neg, reviews):
    """
    Words' features extraction
    Inputs:
        - vocab_pos: vocabulary dictionary from positive reviews containing the frequency of words
        - vocab_neg: vocabulary dictionary from negative reviews containing the frequency of words
        - reviews: list of reviews body text
    Output:
        - X: features' matrix which consists of a row per word in the vocabulary and three columns: 
            bias term + frequency of appearance in positive reviews + frequency of appearance in negative reviews
    """
    
    # Initialise features matrix
    X = np.zeros((len(reviews),3))
    
    # Bias term equal to 1 in matrix first column
    X[:,0] = 1
    
    # Loop through each review
    for i, review in enumerate(reviews):
        # Clean word tokens
        words = get_clean_word_tokens(review)
        # Initialize freq count
        freq_pos = 0
        freq_neg = 0
        # Loop through each word token
        for w in words:
            # Increment positive frequency counts by the frequency of each word found in the vocabulary
            if (w,1) in vocab_pos:
                freq_pos += vocab_pos[(w,1)]
            # Increment negative frequency counts by the frequency of each word found in the vocabulary
            if (w,0) in vocab_neg:
                freq_neg += vocab_neg[(w,0)]
                
        # the sum of the positive freqs corresponds to the second features' column
        X[i,1] = freq_pos
        # the sum of the negative freqs corresponds to the third features' column
        X[i,2] = freq_neg

    return X

# Get all reviews' feature matrix
X = get_reviews_features(vocab_pos, vocab_neg, all_reviews)

# Plot features
plt.figure()
plt.plot(X[0:len(reviews_pos),1], X[0:len(reviews_pos),2], 'g.', label='Positive review')
plt.plot(X[len(reviews_pos)+1:len(y),1], X[len(reviews_pos)+1:len(y),2], 'r.', label='Negative review')
plt.legend()
plt.ylim(0, 5000)
plt.xlim(0, 5000)
plt.ylabel('Sum of word frequencies in Negative vocabulary')
plt.xlabel('Sum of word frequencies in Positive vocabulary')


# LOGISTIC REGRESSION

# Define gradient descent function
def gradient_descent(x, y, theta, alpha, num_iters):
    '''
    Performs gradient descent
    Inputs:
        - x: matrix of features which is (m,n+1)
        - y: corresponding labels of the input matrix x, dimensions (m,1)
        - theta: weight vector of dimension (n+1,1)
        - alpha: learning rate
        - num_iters: number of iterations you want to train your model for
    Outputs:
        - J: the final cost
        - theta: final weight vector
    '''
    m = x.shape[0]
    plt.figure()
    for i in range(0, num_iters):
        
        # sigmoid function
        h = 1 / (1 + np.exp(-np.dot(x, theta))) # or h = sigmoid(z)
        
        # calculate the cost function
        J1 = np.dot(y.transpose(), np.log(h))
        J2 = np.dot(np.transpose(1-y), np.log(1-h))
        J = -(1/m) * (J1 + J2)
        plt.plot(i, J, '.')

        # update the weights theta
        theta_d = (alpha/m) * np.dot(x.transpose(), (h-y))
        theta = theta - theta_d
        
    plt.title("Gradient descent")
    plt.ylabel("Loss")
    plt.xlabel("# iterations")
    J = float(J)
    return J, theta


# Divide train and test sets
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)),test_size=0.33 , random_state=42)   

# Apply gradient descent to find theta logistic regress coeffs
J, theta = gradient_descent(X_train, y_train, np.zeros((3, 1)), 1e-9, 3000)


# TEST ACCURACY

# Define predictions function
def make_predictions(X_test, theta):
    """
    Provide predicted labels for test data set
    Inputs:
        - X_test: test data features matrix
        - theta: logistic regression coefficients
    Output:
        - y_pred: predicted labels
    """
    # Apply sigmoid function to test data with provided 
    # logistic regression coefficients to get the estimated probability
    # for positive (prob>=0.5) or negative (prob<0.5) reviews
    y_pred = 1 / (1 + np.exp(-np.dot(X_test, theta)))
    
    # Transform prediction into 1:positive and 0:negative
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    
    return y_pred

#Get predicted labels
y_pred = make_predictions(X_test, theta)

# Accuracy
accuracy = sum(y_pred.ravel() == y_test.ravel())/len(y_test)
print('accuracy:', accuracy)


# Analyse misclassifications
idx_miss = idx_test[np.array(y_pred != y_test).flatten()]
for idx in idx_miss[0:9]:
    print('Misclassified review example: ------------------------')
    print(all_reviews[idx])
    print('TOKENS:')
    print(get_clean_word_tokens(all_reviews[idx]))
