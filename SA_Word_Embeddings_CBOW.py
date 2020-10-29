# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:35:15 2020

In this script the books review from [1] are used as a corpus to extract word 
embeddings using a Continuous Bag Of Words CBOF model.

Only reviews with 5-star rating and 1-star rating are used in this project.

The CBOW model is implemented from scratch.

The resulting word embeddings are visualised in a 2D plot using the first two 
Principal Components. Interesting relationships between words can be observed 
in this plot. Some examples are:
    - words 'lack' and 'take' are at opposite extremes of PC2
    - some words stems appear close together in the map, such as:
        - 'research' & 'idea'
        - 'cook' & 'enjoy'
        - 'beauti' & 'sens'
        - 'relationship' & 'social'
        
@author: Marina Torrente Rodriguez

References:
    [1] John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, 
        Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. 
        Association of Computational Linguistics (ACL), 2007
        (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

"""

import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from SA_utils import get_reviews_dictionary
from SA_utils import get_vocab_freqs
from SA_utils import get_list_of_token
from SA_utils import get_list_of_words
from SA_utils import get_all_reviews
from SA_utils import get_clean_word_tokens

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

# Get list of unique words from positive and negative vocabularies
min_freq = 5 # minimum frequency filer
vocab = get_list_of_words(get_vocab_freqs(reviews_pos, 1), list(), min_freq)
vocab = get_list_of_words(get_vocab_freqs(reviews_neg, 0), vocab, min_freq)

# Vocabulary size
vocab_size = len(vocab)
print('Final Vocabulary size with a minimum frequency of {} is: {}'.format(min_freq, vocab_size))

## Functions to get input data for CBOW model --------------------------------------

# Build word to index and index to word dictionaries
word2ind = dict()
ind2word = dict()
for i, word in enumerate(vocab):
    word2ind[word] = i
    ind2word[i] = word

# Define function for getting windows of words with C words before and
# after the center word
def get_windows(words, C):
    i = C
    while i < (len(words) - C):
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1

# Define function to generate the one-hot-encoding vector for a word
def word_to_one_hot_vector(word, word2ind, vocab_size):
    one_hot_vector = np.zeros(vocab_size)
    if word in word2ind:
        one_hot_vector[word2ind[word]] = 1
    return np.array(one_hot_vector).T

# Define a function that provide the average of one-hot-vectors of the context words
# surrounding the center word
def context_words_to_avg_one_hot_vectors(context_words, word2ind, vocab_size):    
    context_vectors = [word_to_one_hot_vector(w, word2ind, vocab_size) if w in word2ind else np.zeros(vocab_size) for w in context_words]
    return np.mean(context_vectors, axis=0)

# Define a function that provides the training input set
def get_training_data_batches(data, C, word2ind, vocab_size, batch_size):
    for ind in range(0, len(data), batch_size):
        words = data[ind:min(ind + batch_size, len(data))]
        x = np.zeros((vocab_size,1))
        y = np.zeros((vocab_size,1))
        for context_words, center_word in get_windows(words, C):
            context_vector = context_words_to_avg_one_hot_vectors(context_words, word2ind, vocab_size)
            center_vector = word_to_one_hot_vector(center_word, word2ind, vocab_size)
            x = np.append(x, np.expand_dims(context_vector, axis=1), axis=1)
            y = np.append(y, np.expand_dims(center_vector, axis=1), axis=1)
        yield x[:,1:len(x)], y[:,1:len(x)]


## Function to compute the CBOW model ---------------------------------------------

# Activation functions
def ReLU(z):
    return np.maximum(0, z)

def Softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 0)

# Initialise neural network weights and biases with random numbers
def initialize_model(emb_size, vocab_size, random_seed=1):
    np.random.seed(random_seed)
    W1 = np.random.rand(emb_size, vocab_size)
    W2 = np.random.rand(vocab_size, emb_size)
    b1 = np.random.rand(emb_size, 1)
    b2 = np.random.rand(vocab_size, 1)
    return W1, W2, b1, b2

# Define forward propagation function
def forward_propagation(x, W1, W2, b1, b2):
    h = ReLU(np.dot(W1,x) + b1)
    z = np.dot(W2,h) + b2
    return z, h

# Define cost entropy cost function
def cross_entropy_cost(y, yhat, batch_size):
    logprobs = np.multiply(np.log(yhat),y) + np.multiply(np.log(1 - yhat), 1 - y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

# Define backwards propagation function
def back_propagation(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    r1 = ReLU(np.dot(W2.T, yhat - y))
    # gradient of W1
    grad_W1 = (1/batch_size) * np.dot(r1, x.T)
    # gradient of W2
    grad_W2 = (1/batch_size) * np.dot(yhat-y, h.T)
    # gradient of b1
    grad_b1 = (1/batch_size) * np.sum(r1, axis=1, keepdims=True)
    # gradient of b2
    grad_b2 = (1/batch_size) * np.sum(yhat-y, axis=1, keepdims=True)
    
    return grad_W1, grad_W2, grad_b1, grad_b2

# Define gradient descent function
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    
    # Initialize weights and biases
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=1234)
    
    # Define data input processing parameters 
    batch_size = 2*128  # Size of word batches
    C = 2              # Window size = 2*C + 1 

    # Initialize iterations count
    iters = 0
    
    # Update neural network parameters based on data batches
    for x, y in get_training_data_batches(data, C, word2ind, vocab_size, batch_size):

        # Forward propagation through shallow neural network
        z, h = forward_propagation(x, W1, W2, b1, b2)

        # Get center word vector estimate
        yhat = Softmax(z)

        # Get cost
        cost = cross_entropy_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")

        # Get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_propagation(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        
        # Update weights and biases with gradients
        W1 = W1 - alpha*grad_W1
        W2 = W2 - alpha*grad_W2
        b1 = b1 - alpha*grad_b1
        b2 = b2 - alpha*grad_b2
        
        # Increment iterations count
        iters += 1 
        
        # Stop processing if max number of iterations achieved
        if iters == num_iters: 
            break
        
        # Update alpha every 10 iterations
        if iters % 10 == 0:
            alpha *= 0.66
            
    return W1, W2, b1, b2


## Extract Word embeddings --------------------------------------------------------

# Get word token's list from positive reviews
data = get_list_of_token(reviews_pos, list())
data = get_list_of_token(reviews_pos, data)

# shuffle reviews randomly to mix positive and negative reviews
import random
random.shuffle(data)

# Flatten word tokens list 'data'
data = [val for sublist in data for val in sublist]

# Define length of embeddings N
N = 300

# Calculate neural network weights
num_iter = 150
W1, W2, b1, b2 = gradient_descent(data, word2ind, N, vocab_size, num_iter)

# Word embeddings
word_embeddings = (W1.T + W2)/2.0



## Visualise using first 2 Principal Components -----------------------------------
pca = PCA(n_components=2)
result = pca.fit_transform(word_embeddings)
plt.figure(figsize=(16,7))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(vocab):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Word embeddings visualisation in the two first Principal Components")
plt.show()


## Classification model using word embeddings
# Get all reviews body text in a list and corresponding positive/negative labels
all_reviews, y = get_all_reviews(reviews_pos, reviews_neg)

# Iterate through reviews and get the sum vector from the word tokens in the review
sum_vector = np.zeros((len(all_reviews), N))
for i, review in enumerate(all_reviews):
    words = get_clean_word_tokens(review)
    vector = [word_embeddings[word2ind[word],:] for word in words if word in word2ind]
    sum_vector[i,:] = np.sum(vector, axis=0).T

# Classification model using extreme gradient boosting (XGBoost)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# XBG model
model = XGBClassifier(objective='binary:logistic', booster='gbtree')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, sum_vector, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print("XGBoost model mean accuracy: {}".format(scores.mean()))


