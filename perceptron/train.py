#Training function
import numpy as np
import copy
import time
from .utils import loss, accuracy, data_shuffle
from .gradient import batch_gradient
from .three_layer_perceptron import ThreeLayerPerceptron

def training(train_data, train_labels, dev_data, dev_labels, \
             learning_rate, momentum, layer2, layer3, precision): 
    """
    Trains a 3-layer MLP using the training set, using the dev
    loss as a stopping condition, using the batch_gradient function.
    
    Inputs:  train_data, train_labels, dev_data, dev_labels,
             learning_rate, momentum, layer2 (size), layer3 (size),
             precision (stopping condition)
            
    Outputs: train_acc, train_loss, dev_acc, dev_loss, time_elapsed
             epoch (number of), m (the trained model)
    """
    #Initialize perceptron
    m = ThreeLayerPerceptron(784, layer2, layer3)
    
    #Initialize output lists
    train_acc = [accuracy(m, train_data, train_labels)]
    train_loss = [loss(m, train_data, train_labels)]
    dev_acc = [accuracy(m, dev_data, dev_labels)]
    dev_loss = [loss(m, dev_data, dev_labels)]
    
    #Initialize velocities, for introducing a momentum term
    v_W1 = 0.0
    v_W2 = 0.0
    v_w3 = 0.0
    v_b1 = 0.0
    v_b2 = 0.0
    v_b3 = 0.0
    
    current_loss = loss(m, train_data, train_labels)
    start = time.time()
    epoch = 0
    
    while True: 
        epoch += 1
        
        #Shuffle data each epoch
        train_data, train_labels =  \
        data_shuffle(train_data, train_labels) 
        
        #Get gradient of batch
        gradm = batch_gradient(m, train_data, train_labels)     
        
        #Update velocities
        v_b1 = v_b1*momentum - learning_rate*(gradm.b1)
        v_W1 = v_W1*momentum - learning_rate*(gradm.W1)
        v_b2 = v_b2*momentum - learning_rate*(gradm.b2)
        v_W2 = v_W2*momentum - learning_rate*(gradm.W2)
        v_b2 = v_b3*momentum - learning_rate*(gradm.b3)
        v_w2 = v_w3*momentum - learning_rate*(gradm.w3)
    
        #Update weights and biases
        m.b1 += v_b1
        m.W1 += v_W1
        m.b2 += v_b2
        m.W2 += v_W2
        m.b3 += v_b3
        m.w3 += v_w3
        
        #Create output lists
        new_loss = loss(m, train_data, train_labels)
        train_acc.append(accuracy(m, train_data, train_labels))
        train_loss.append(new_loss)
        dev_acc.append(accuracy(m, dev_data,dev_labels))
        dev_loss.append(loss(m, dev_data,dev_labels))
        
        #Determine when to exit.
        if abs(new_loss - current_loss) < precision:
            break
        current_loss = new_loss
        
        if epoch%50==0:
            print(f'Epoch: {epoch}')
    
    #Determine runtime
    finish = time.time()
    time_elapsed = finish - start
    
    print(f"Converged after {time_elapsed} s and {epoch} epochs.")
    
    return train_acc, train_loss, dev_acc, dev_loss, \
            time_elapsed, epoch, m
