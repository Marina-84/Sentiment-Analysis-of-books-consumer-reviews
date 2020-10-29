# Sentiment-Analysis-of-books-consumer-reviews

NLP techniques have been implemented to analyse the consumer reviews of books.

The data set used in this project is found in https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

The different methods are outlined next:
* Sentiment classifier with Logistic Regression
* Sentiment classifier with Naive Bayes and visualisation of words frequency versus positive to negative probability ratio
* Word embeddings using CBOW and PCA to visualise word relationships

Prior to any analysis the words are pre-process. The pre-procesing of the body text of the reviews consists of the following steps:
1. Removing digits
2. Lowercasing
3. Removing punctuation
4. Tokenizing words
5. Remove stop words
6. Remove the word 'books' which frequency is much higher than the rest of words and can skew the results
7. Removing stop words.py
8. Stemming

The structure of the project is organized as follows:
* 'SA_utils.py': contains useful processing function common for the different methods of analysis
* 'SA_Logistic_Regression_classification.py': is a python script detailing the steps to develop a sentiment classifier using Logistic Regression where the algorithm is implemented from scratch
* 'SA_Naive_Bayes_classification.py': is a python script containing the implementation of a naive bayes classifier also implemented step by step. The negativity of positivity of words is visualise against their frequency using the ratio between the probability of words appearing in positive reviews and the probability of words appearing in negative reviews.
* 'SA_Word_Embeddings_CBOW.py': is a python script where the Continuous Bag of Words method to generate word vectors known as word embeddings has been implemented. A Principal Components Analysis is applied to the embeddings and the two first PCs are used for visualising relationships between words in as distances in a 2D map.

Each of the NLP techniques used in this project is further explained in the following sections.

