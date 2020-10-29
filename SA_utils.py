# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:29:00 2020

Collection of useful functions to handle text data

@author: Marina Torrente Rodriguez
"""

import re
from nltk.tokenize import RegexpTokenizer  # module for tokenizing removing punctuation
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
import numpy as np


# Organise reviews from files into a dictionary
def get_reviews_dictionary(file):
    """
    Gets the information from the text file of consumer reviews and returns a 
    structured dictionary of the reviews
    
    Input:
        - file: books reviews file
    Output:
        - reviews: dictionary where the key is each review unique identifier and 
        contains a 4-element tuple with (Book's title, Book's author, rating, body text) 
    """
    
    # Initialise empy list where data from reviews will be stored
    asin_idx = list()       # For identifiers
    prod_name_idx = list()  # For product names, in this case book title + author
    rating_idx = list()     # For consumer ratings in 5-stars style line index
    revtext_idx = list()    # For reviews line index
    revtexts = list()       # For reviews body text
            
    # Create a dictionary where keys are the unique id of reviews
    reviews = dict()        # Output dictionary containing organised reviews' data
    for i, line in enumerate(file): 
        
        # ID (asin)
        if "<asin>" in line:        # If <asin> heading is found, append identifier to list from the next line
            asin_idx.append(i+1)
        if i in asin_idx: 
            asins = line[0:10]
         
        # Product name
        if "<product_name>" in line:# If <product_name> heading is found, get product name to list from the next line
            prod_name_idx.append(i+1)
        if i in prod_name_idx:
            prod_names = line
            prod_names_split = prod_names.split(': Books:') # Split in title (before ':Books' pattern) and author (after ':Books' pattern)
            book_title = prod_names_split[0]    # Save book title
            if len(prod_names_split)>1:         # Save author if provided (some reviews don't provide author's name)
                book_author = prod_names_split[1]
            else:
                book_author = 'NaN'
    
        # Rating
        if "<rating>" in line:      # If <rating> heading is found, append rating to list from the next line
            rating_idx.append(i+1)
        if i in rating_idx:
            ratings = int(line[0])
            
        # Text
        if "<review_text>" in line: # If <review_text> heading is found, append review's text body to list from the next line
            revtext_idx.append(i+1)
            revtexts = list()       # Initialise list: required for every review due to multiple lines option available
        if i in revtext_idx and "</review_text>" not in line:
            revtexts.append(line)
            revtext_idx.append(i+1)
        elif revtexts:  # Once the text body has been stored the review is ready to be stored in the dictionary
            # fill dictionary input
            s = "\n"
            s = s.join(revtexts) 
            if ratings == 1 or ratings == 5: # Only 1-star or 5-star rating reviews are saved
                reviews[asins] = (book_title, book_author, ratings, s)
        
    return reviews


# Define function to pre-process reviews text
def get_clean_word_tokens(rev_text):
    """
    Transforms the text of the reveiws into lowercase word tokens, with no punctuation,
    no digits and stemmed
    
    Input:
        - rev_text: text body of the consumer reviews
    Output:
        - stem_words: tokenized clean words
    """
    
    # Remove numbers
    rev_text = re.sub(r'[0-9]', '', rev_text)
    
    # Lowercase
    rev_text = rev_text.lower()
    
    # Tokenize the string removing punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(rev_text)
    
    # Remove stop words
    stopwords_english = stopwords.words('english') 
    stopwords_english.append('book')
    stopwords_english.append('books')
    
    clean_words = [w for w in words if not w in stopwords_english] 
    
    # transform words into their stems: stemming
    stemmer = PorterStemmer()
    stem_words = [stemmer.stem(w) for w in clean_words]

    return stem_words


# Create vocabulary with frequency counts 
def get_vocab_freqs(reviews, sentiment):
    """
    Provides a vocabulary dictionary containing the words' frequencies
    
    Inputs:
        - reviews: dictionary of reviews generated using the function 'get_reviews_dictionary'
        - sentiment: integer specifying positive (value of 1) or negative (value of 0) reviews
    Output:
        - freqs: dictionary where the keys are tokenized words from the reviews' text body and the
        value correspond to the frequency of appearance of the corresponding word
    """
    
    # Initialize dict
    freqs = {}
    
    # Loop through reviews
    for key in reviews.keys():
        # loops through word tokens and count frequency
        words = get_clean_word_tokens(reviews[key][3])        
        for w in words:
            if (w,sentiment) in freqs: # Increase frequency if key already in dict
                freqs[(w,sentiment)] += 1
            else:                       # Initialize freq count if key not in dict yet
                freqs[(w,sentiment)] = 1
    if sentiment == 1:
        print('The size of vocabulary from positive reviews is ', len(freqs))
    else:
        print('The size of vocabulary from negative reviews is ', len(freqs))
                
    # Remove words that appear less than the minimum frequency defined
    min_freq = 5
    remove_key = list()
    for i, key in enumerate(freqs):
        if freqs[key] < min_freq:
            remove_key.append(key)
    for key in remove_key:
        del freqs[key]
    print("The size of vocabulary filtered by minimum frequency of {} is {}".format(min_freq, len(freqs)))

    return freqs


# Define a function to join positive and negative reviews in a list
def get_all_reviews(reviews_pos, reviews_neg):
    """
    Get reviews dictionaries and returns a list including only the body text and corresponding positive or 
    negative labels
    
    Inputs:
        - reviews_pos: dictionary of positive reviews generated using the function 'get_reviews_dictionary'
        - reviews_neg: dictionary of negative reviews generated using the function 'get_reviews_dictionary'
    Output:
        - reviews: list with positive and negative reviews' body text
        - y: reviews labels vector with 1's and 0's indicating positive and negative reviews position in the list respectively
    """
    
    # Initialize a list to store positive and negative reviews
    all_reviews = list()
    
    # Iterate through positive reviews storing the text body in the all_reviews list
    for key in reviews_pos.keys():
        all_reviews.append(reviews_pos[key][3])

    # Iterate through negative reviews storing the text body in the all_reviews list
    for key in reviews_neg.keys():
        all_reviews.append(reviews_neg[key][3])

    # vector with 1's and 0's indicating positive and negative reviews position in the list respectively 
    y = np.zeros((len(all_reviews),1))
    y[0:len(reviews_pos)] = 1
    
    return all_reviews, y


# Define a function to get a list of words from a vocabulary of words frequencies
# with a minimum frequency filter
def get_list_of_words(freqs, vocab, min_freq):
    """
    Input:
        - freqs: vocabulary dictionary of keys corresponding to words and 
        values to their frequencies
        - vocab: empty or existing vocabulary list
        - min_freq: integer specifying the minimum frequency of words required 
        to be added onto the output vocabulary list
    Output:
        - vocab: vocabulary list where words with a minimum frequency defined 
        in min_freq are added
    """
    for key in freqs.keys():
        word, sentiment = key
        if word not in vocab and freqs[key] > min_freq:
            vocab.append(word)
    return vocab


# Define a function to get all reviews as a list of clean word tokens
def get_list_of_token(reviews, data):
    """
    Inputs:
        - reviews: dictionary of reviews generated using the 
        function 'get_reviews_dictionary'
        - data: empty or existing list of word tokens sequence where the 
        new sequence of tokens from 'reviews' will be added
    Output:
        - data: list of word tokens where the sequence of tokens from the 
        reviews dictionary are added
    """
    for key in reviews.keys():
        data.append(get_clean_word_tokens(reviews[key][3]))
    return data
