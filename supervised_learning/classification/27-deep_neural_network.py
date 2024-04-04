#!/usr/bin/env python3
"""
Deep Neural Network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """
    Class that represents a Deep Neural Network
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        Gets the private instance attribute __L
        """
        return (self.__L)

    @property
    def cache(self):
        """
        Gets the private instance attribute __cache
        """
        return (self.__cache)

    @property
    def weights(self):
        """
        Gets the private instance attribute __weights
        """
        return (self.__weights)

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """

        # store X in A0
        if 'A0' not in self.__cache:
            self.__cache['A0'] = X
        
        for i in range(1, self.__L + 1):
            # First layer
            if i == 1:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                # Multiplication of weight and add bias
                z = np.matmul(W, X) + b
            else:  # Next layers
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                X = self.__cache['A{}'.format(i - 1)]
                Z = np.matmul(W, X) + b
        
            # Activation function :
            # For last use softmax for multiclass
            if i == self.__L:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache["A{}".format(i)] = (
                        exp_Z / np.sum(exp_Z, axis=0, keepdims=True))
            else:
                self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i)], self.__cache
            
    def cost(self, Y, A):
        """
        Calculate cross-entropy cost for multiclass
        """

        # Store m value
        m = Y.shape[1]

        # Calculate log loss function
        log_loss = -(1 / m) * np.sum(Y * np.log(A))
        
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        
        # Run forward propagation
        output, cache = self.forward_prop(X)

        # Calculate cost
        cost = self.cost(Y, output)

        # Convert predicted proba to one-hot
        result = np.zeros_like(output)

        # Label values
        result[np.argmax(output, axis=0), np.arange(output.shape[1])] = 1

        return result, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculate the gradient descent
        """
        m = Y.shape[1]
        back = {}
        for index in range(self.L, 0, -1):
            A = cache["A{}".format(index - 1)]
            if index == self.L:
                back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            else:
                dz_prev = back["dz{}".format(index + 1)]
                A_current = cache["A{}".format(index)]
                back["dz{}".format(index)] = (
                    np.matmul(W_prev.transpose(), dz_prev) *
                    (A_current * (1 - A_current)))
            dz = back["dz{}".format(index)]
            dW = (1 / m) * (np.matmul(dz, A.transpose()))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            W_prev = self.weights["W{}".format(index)]
            self.__weights["W{}".format(index)] = (
                self.weights["W{}".format(index)] - (alpha * dW))
            self.__weights["b{}".format(index)] = (
                self.weights["b{}".format(index)] - (alpha * db))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        trains the neuron and updates __weights and __cache
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        saves the instance object to a file in pickle format
        """
        import pickle
        if not isinstance(filename, str):
            print("Error: filename must be a string.")
            return

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        loads a pickled DeepNeuralNetwork object from a file
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None
