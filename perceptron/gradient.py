#Functions for calculating gradients and batch gradients
import numpy as np
import copy
import time
from .three_layer_perceptron import zero
from .utils import loss, logistic, f

def flatten_weights(m):
    """
    Input:  m - perceptron model
    
    Returns: an array with a list of flattened weights and biases
             in the order w1b1w2b2 ...
    """
    flat = np.concatenate([m.W1.flatten(), m.b1.flatten(), \
                           m.W2.flatten(), m.b2.flatten(), \
                           m.w3.flatten(), [m.b3]])
    return flat


def unflatten_weights(m, weights):
    """
    Inputs:  m - perceptron model
             weights - vector of all flattened weights/biases
             
    Returns: a ThreeLayerPerceptron object with the unflattened
             weights/biases
    """
    gradm = zero(m)
    b1_index = m.W1.size
    W2_index = m.W1.size+m.b1.size
    b2_index = m.W1.size+m.b1.size+m.W2.size
    w3_index = m.W1.size+m.b1.size+m.W2.size+m.b2.size
    gradm.W1 = weights[:b1_index].reshape(m.W1.shape)
    gradm.b1 = weights[b1_index:W2_index].reshape(m.b1.shape)
    gradm.W2 = weights[W2_index:b2_index].reshape(m.W2.shape)
    gradm.b2 = weights[b2_index:w3_index].reshape(m.b2.shape)
    gradm.w3 = weights[w3_index:-1].reshape(m.w3.shape)
    gradm.b3 = weights[-1]
    
    return gradm
    
    
def finite_difference_gradient(m, x, y):
    """
    Calculates and outputs all gradients for a single sample x/y
    using finite difference.
    
    """
    epsilon = 0.00000001
    weights = flatten_weights(m)
    gradients = list()
    for num in range(len(weights)):
        weights[num] += epsilon/2.0
        l = loss(unflatten_weights(m, weights), x, y)
        weights[num] -= epsilon
        r = loss(unflatten_weights(m, weights), x, y)
        gradients.append((l-r)/epsilon)
    gradm = unflatten_weights(m, np.asarray(gradients))
    return gradm


def batch_gradient(m, X, Y):
    """
    Gets gradients of a whole batch at once.
    
    """
    gradm = zero(m)
    
    #Forward pass. l2 = layer 2 size, l3 = layer 3 size
    h1 = logistic(np.dot(m.W1, X).T + m.b1) #(12000,l2) array
    h2 = logistic(np.dot(h1, m.W2.T) + m.b2) #(12000,l3) array
    h3 = f(m, X) #(12000,) array
    
    #Backwards pass for w3 and b3
    error3 = h3 - Y #(12000,) array
    gradm.b3 = np.sum(error3)/ Y.size #scalar
    #Now get (l3,12000) array and sum columns
    gradm.w3 = np.sum(h2.T*error3, axis = 1) / Y.size #(l3,) array  
    
    #Backwards pass for w2 and b2
    #Multiply h2 and (1-h2) elementwise for (12000,l3) array
    #Then multiply elementwise by row by w3 to get (12000,l3)
    factor2 = (h2*(1-h2))*m.w3 #(12000, l3) array
    #Transpose, then row-wise mult = (l3, 12000) array, 
    #then sum across columns for (l3, 12000) array
    gradm.b2 = np.sum(factor2.T * error3, axis = 1) / Y.size 
    #For W2, first multiply factor2.T by error 3 to get a (l3,12000) 
    #array. Then get elementwise outer products for 12000 columns 
    #of factor2.T*error3 and h1 and add these. 
    #This can be done in numpy by dotting these matrices
    gradm.W2 = np.dot(factor2.T*error3, h1) / Y.size #(l3, l2) array

    #Backwards pass for w1 and b1
    #h2*(1-h2)*w3 for (12000, l3) array. Then matrix mult by (l3, l2)
    #W2 to get (12000,l2) array. Then mult elementwise by h1*(1-h1)
    factor1 = np.matmul((h2*(1-h2))*m.w3, m.W2)*(h1*(1-h1)) #(12000,l2)
    #Mult factor1.T by error for (l2,12000) array. Then sum cols.
    gradm.b1 = np.sum(factor1.T * error3, axis = 1) / Y.size #(l2,) 
    #Similar process to above for outer product. Dot (l2, 12000) mat
    #and (12000, 784) mat, for (l2, 784) matrix. 
    gradm.W1 = np.dot(factor1.T*error3, X.T) / Y.size
    
    return gradm

