import numpy as np
import copy
import time

class ThreeLayerPerceptron: 
    """
    3-layer perceptron model class. This is updated during training
    with weights W and biases b.
    """
    def __init__(self, in_size, layer_2_size, layer_3_size):
        np.random.seed(1)
        #Glorot initialization
        x3 = np.sqrt(6/(layer_3_size+1))*4
        x2 = np.sqrt(6/(layer_2_size+layer_3_size))*4
        x1 = np.sqrt(6/(layer_2_size + in_size))*4
        self.W1 = -2*x1*np.random.rand(layer_2_size, in_size) + x1
        self.W2 = -2*x2*np.random.rand(layer_3_size, layer_2_size) + x2
        self.w3 = -2*x3*np.random.rand(layer_3_size) + x3
        self.b1 = np.zeros(layer_2_size)
        self.b2 = np.zeros(layer_3_size)
        self.b3 = float(0.0)

              
def zero(m):
    """
    Input:   m - perceptron model
    
    Returns: an object like model m but initialized to zero
    """
    gradm = copy.deepcopy(m)
    gradm.W1 = np.zeros(m.W1.shape)
    gradm.W2 = np.zeros(m.W2.shape)
    gradm.w3 = np.zeros(m.w3.shape)
    gradm.b1 = np.zeros(m.b1.shape)
    gradm.b2 = np.zeros(m.b2.shape)
    gradm.b3 = float(0.0)
    return gradm


