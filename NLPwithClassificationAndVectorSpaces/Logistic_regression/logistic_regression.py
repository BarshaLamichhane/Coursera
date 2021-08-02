##https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a

import nltk                                  # Python library for NLP
from nltk.corpus import twitter_samples      # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt              # visualization library
import numpy as np  
from numpy.random import rand
import random
from math import sqrt


np.random.seed(300)

# download the stopwords for the process_tweet function
nltk.download('stopwords')

# import our convenience functions
from utils import process_tweet, build_freqs
# process_tweet, build_freqs

# select the lists of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set) 
""""combine positive and negative labels
# np.ones() - create an array of 1's
# np.zeros() - create an array of 0's
# np.append() - concatenate arrays
# concatenate the lists, 1st part is the positive tweets followed by the negative"""
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)


frequencies = build_freqs(train_x, train_y)
test_freqs = build_freqs(test_x, test_y)


        
# print(data)

# x= np.zeros((1,3))
# print(x)

def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        if (word, 1) in freqs:
            x[0,1] += freqs[(word, 1)]
        
        # increment the word count for the negative label 0
        if (word, 0) in freqs:
            x[0,2] +=freqs[(word, 0)]
        
    ### END CODE HERE ###
    assert(x.shape == (1, 3))
    return x


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # calculate the sigmoid of z
    h = 1.0/(1.0 + np.exp(-z))
    ### END CODE HERE ###
    
    return h

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    m = x.shape[0]
    print("x_shape is",m)
    
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        # z = None
        
        z = np.dot(x, theta)
        z = z.reshape(m, -1)
        
        
        # y = np.array(y)
       
        # get the sigmoid of z
        h = sigmoid(z)
        h = np.array(h)
       

        J = -(1 / m)*(np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))
       

        # update the weights theta
        theta = theta - alpha*(np.dot(x.T, (h-y)))/m   
    ### END CODE HERE ###
    return J, theta ## final J and theta

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def normalize_input(X, a=-1, b=1):
    # Normalizing the input
    X = a + ((X - np.min(X)) * (b - a) / (np.max(X) - np.min(X))) 
    return X

def get_all_features(tweet, freq):
    """
    tweet strimg must be in array, otherwise while taking length of tweet it will count all the characters of string

    """
    """collect the features 'x' and stack tehm into matrix 'X'"""

    X = np.zeros((len(tweet), 3),dtype=np.float64)

    for i in range(len(tweet)):
        X[i, :] = extract_features(tweet[i], freq)

    X = normalize_input(X)
    
    return X

def train(x_train, y_train, freqs, ):
    """collect the features 'x' and stack tehm into matrix 'X'"""
    X = get_all_features(x_train, freqs)
    Y =  y_train
    initial_weights = np.random.randn(3,1)
    xaiver_weights = initial_weights * np.sqrt(1/3)
    J, theta = gradientDescent(X, Y, xaiver_weights, 0.001, 1500)
    # print(J)
    # print(theta)
    return J, theta




def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # extract the features of the tweet and store it into x
    x = get_all_features(tweet, freqs)
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))
    
    ### END CODE HERE ###
    return y_pred    



# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def test_logistic_regression(x_test, y_test, freqs, theta):
    
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # the list for storing predictions
    y_hat = []
    y_test = y_test.flatten()
   
    
    
    for tweet in x_test:
        # tweet = [tweet]
        # get the label prediction for the tweet
        y_pred = predict_tweet([tweet], freqs, theta)
        if y_pred >= 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)
    accuracy = np.sum(y_hat == y_test) / len(y_test)
    # .all(axis=0).mean()
    print("Accuracy is: ",accuracy)

    ### END CODE HERE ###
    
    return accuracy



# # Some error analysis done for you

def error_analysis(x_test, y_test, theta):
    no_of_miclassification = []
    for x,y in zip(x_test,y_test):
        y_hat = predict_tweet([x], test_freqs, theta)
        if(np.abs(y - (y_hat >= 0.5)) > 0):
            no_of_miclassification.append(y_hat)
            print('THE TWEET IS:', x)
            print('THE PROCESSED TWEET IS:', process_tweet(x))
            print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))
    # print("Total no of misclassified sentiments :",len(no_of_miclassification))



def predict_own_tweet():

    my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
    # print(process_tweet(my_tweet))
    y_hats = predict_tweet([my_tweet], frequencies, thetas)
    # print(y_hats)
    if y_hats > 0.5:
        print(f'{my_tweet} is Positive sentiment')
    else: 
        print(f'{my_tweet} is Negative sentiment')
    


if __name__ == '__main__':
    J, thetas = train(train_x,train_y,frequencies)
    test_logistic_regression(test_x, test_y, test_freqs, thetas)
    # error_analysis(test_x, test_y, thetas)
    predict_own_tweet()