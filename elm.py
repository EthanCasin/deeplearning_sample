import numpy as np
from scipy.linalg import pinv2 # moore-penrose pseudo-inverse 

class ELM(object):
    """
    Creates an Extreme Learning Machine model.
    
    Extreme Learning Machine (ELM) models are known to be very fast shallow
    feed-forward networks. The weights are initially created randomly from a
    distribution (typically from a normal distribution). For this project, a
    uniform distribution is used by defauly because of the scaled data.
    Additionally, instead of relying on backpropagation, ELM models utilize the 
    moore-penrose inverse in setting the weights. 
    
    
    Parameters:
    -----------
    n_visible: int
        number of visible units to be created. It is the number of 
        features in the dataset

    n_hidden: int
        number of hidden nodes to be set. 

    distribution: string or np.random distribution
        The distribution where the weights and biases are initially 
        generated

    Methods:
    -----------
    .fit : array
        Fits the data to the model
    .predict : array
        Predicts the output of a given input
    """
        
    def __init__(self, n_visible, n_hidden, distribution='uniform'):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.distribution = distribution
        
        # initialize weights and biases based on set distribution
        if self.distribution == 'uniform':
            self.weight = np.random.uniform(size=[self.n_visible, self.n_hidden])
            self.bias = np.random.uniform(size=[self.n_hidden])
        elif self.distribution == 'normal':
            self.weight = np.random.normal(size=[self.n_visible, self.n_hidden])
            self.bias = np.random.normal(size=[self.n_hidden])

        # for user defined distributions based on np.random
        else: 
            self.weight =  self.distribution(size=[self.n_visible, self.n_hidden])
            self.bias =  self.distribution(size=[self.n_hidden])
        self.beta = np.array([])
        self.activation = ''
        
    def sigmoid(self, x):
        """Returns the output of a sigmoid function."""
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        """Returns the output of a ReLu function."""
        return np.maximum(x, 0, x)
    
    def tanh(self, x):
        """Returns the output of a tanh function."""
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def hidden_nodes(self, X):
        """Propagates the data from input to output layers through the hidden nodes.
        
        Parameter:
        ----------
        X : array-like
            Input data to be fitted or predicted.
        """
        X = np.array(X) # make the input as an array
        # compute the inner terms of g (i.e. w * x + b)
        # recall that g(w * x + b)
        G = X.dot(self.weight) + self.bias 

        # compute the hidden layer output matrix H
        # based on the activation function
        # the activation function is retrieved from the .fit() method
        if self.activation == 'relu':
            return self.relu(G)
        elif self.activation == 'sigmoid':
            return self.sigmoid(G)
        else:
            return self.tanh(G)
        
    def fit(self, X, y, activation='relu'):
        """Fits the training data to the model based on an activation function.
        
        Parameters:
        -----------
        X : array-like
            The input data to be fit by the model.

        y : array-like
            The targets of the data. This is the target matrix T

        activation: string
            The selected activation function. Options are:
            'relu', 'sigmoid', 'tanh'
        
        Returns:
        --------
        Beta : array
            The learned weights.
        """
        # convert X and y to arrays
        X = np.array(X)
        y = np.array(y)

        # set the activation function for the whole ELM object
        self.activation = activation

        # compute the output weight vectors (beta)
        # This is retrieved using the Moore-Penrose generalized inverse
        self.beta = np.dot(pinv2(self.hidden_nodes(X)), y)
        return self.beta
        
    def predict(self, X):
        """Predicts the output of X. Use the validation or test set for this one.

        Note that the activation function can output continuous values after predicting.
        There are two work-arounds: use np.rint(y) or np.where(y > 0.5, 1, 0). The latter
        is based on the sigmoid cut-off, but np.rint() normally works fine. 

        Parameters:
        -----------
        X : array-like
            Validation or test set to be predicted

        Returns:
        --------
        y : array
            The predictions in array form. 
        """
        X = np.array(X) # convert input to array

        # propagate input to the network
        output = self.hidden_nodes(X)

        # compute the output
        # T = H * Beta
        return output.dot(self.beta)
    