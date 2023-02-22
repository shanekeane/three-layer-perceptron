#Useful functions
import numpy as np
import copy

def logistic(x):
    """
    Logistic function
    """
    return 1.0/(1.0 + np.exp(-x))


def f(m, X):
    """
    Prediction function
    
    Inputs:  m - perceptron model
             x - sample inputted to the model
            
    Returns: The results of applying the logistic function
             when putting the sample through the model
    """
    #Take tranpose to get 12000xlayer_2_size array to
    #enable m.b1 to be added
    h1 = logistic(np.matmul(m.W1, X).T + m.b1)
    #Again take transpose of W2 so that we have 12000xlayer2
    #times layer3xlayer2 size.
    h2 = logistic(np.matmul(h1, m.W2.T) + m.b2)
    #Outputs shape (12000,) array
    return logistic(np.dot(h2, m.w3) + m.b3)


def p(m, X):
    """
    Returns a prediction given model and sample
    
    Inputs:  m - perceptron model
             x - sample inputted to the model
            
    Returns: either a 1 or 0 predicting which class
    
    """
    return f(m, X) >= 0.5


def accuracy(m, X, Y):
    """
    Inputs:  m - perceptron model
             X - dataset
             Y - labels for the dataset
            
    Returns: the accuracy of the dataset
    """
    return sum(p(m, X) == Y)*100/len(Y)
         
    
def loss(m, X, Y):
    """
    Inputs:  m - perceptron model
             X - dataset
             Y - labels for the dataset
            
    Returns: the loss of the dataset
    """
    #For entire batch
    if np.ndim(X) > 1:
        return (- np.dot((1-Y), np.log(1-f(m, X))) \
                - np.dot(Y, np.log(f(m, X))))/len(Y)
    #For single sample.
    else:
        return -(1-Y)*np.log(1-f(m, X)) - y*np.log(f(m, X))


def data_shuffle(data, label):
    """
    Returns a shuffled version of data and labels. 
    """
    order = np.arange(len(label))
    np.random.shuffle(order)
    return data[:, order], label[order]
