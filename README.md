# three-layer-perceptron
This provides utilities to enable one to train a three layer perceptron with inputted training and validation data sets. 

# Description 
- The three-layer perceptron and all associated functions are coded from scratch, and enable one to specify layer sizes for two layers of the perceptron.
- The training process uses function "training", imported from perceptron.train. This takes the training/validation data and labels as inputs, as well as perceptron layers sizes and other training parameters.
- The demo notebook shows the use of this function.
- The dataset provided is a version of the fashion MNIST dataset given here: https://github.com/zalandoresearch/fashion-mnist. It has been simplified to only 12,000 training and 1,000 validation samples, and the labels define a binary classification problem, specifying whether the image is of a bag or not. 

# Packages
Works with:
- Python 3.9.7
- Numpy 1.20.3
