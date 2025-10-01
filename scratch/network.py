import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self, x_train):
        '''
        Forward propagation algorithm.
        Arg: Returns the output of the network.
        '''
         # Store parameters for easier access
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3 = self.params['W3']
        
        # Layer 1: input -> hidden1
        self.Z1 = np.dot(W1, x_train)
        self.A1 = self.activation_func(self.Z1)
        
        # Layer 2: hidden1 -> hidden2
        self.Z2 = np.dot(W2, self.A1)
        self.A2 = self.activation_func(self.Z2)
        
        # Layer 3: hidden2 -> output
        self.Z3 = np.dot(W3, self.A2)
        self.A3 = self.output_func(self.Z3)
        
        return self.A3



    def _backward_pass(self, y_train, output):
        '''
        Backpropagation algorithm responsible for updating the weights of the neural network.

        Args:
        Returns a dictionary of the weight gradients which are used to 
        update the weights in self._update_weights().

        '''
        m = y_train.shape[0]  
        
        # Store current input
        x_train = self.x_current
        
        # Output layer gradient
        dZ3 = output - y_train
        dW3 = np.outer(dZ3, self.A2)
        
        # Hidden layer 2 gradient
        dA2 = np.dot(self.params['W3'].T, dZ3)
        dZ2 = dA2 * self.activation_func_deriv(self.Z2)
        dW2 = np.outer(dZ2, self.A1)
        
        # Hidden layer 1 gradient
        dA1 = np.dot(self.params['W2'].T, dZ2)
        dZ1 = dA1 * self.activation_func_deriv(self.Z1)
        dW1 = np.outer(dZ1, x_train)
        
        return {
            'W1': dW1,
            'W2': dW2,
            'W3': dW3
        }


    def _update_weights(self, weights_gradient, learning_rate):
        '''
        Update the network weights according to stochastic gradient descent.
        '''
        self.params['W1'] -= learning_rate * weights_gradient['W1']
        self.params['W2'] -= learning_rate * weights_gradient['W2']
        self.params['W3'] -= learning_rate * weights_gradient['W3']
        


    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        Implement the prediction making of the network.
        Returns the index of the most likeliest output class.

        Arg: returns index of max value
        '''
        # Perform forward pass
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3 = self.params['W3']
        
        # Forward propagation
        Z1 = np.dot(W1, x)
        A1 = self.activation_func(Z1)
        
        Z2 = np.dot(W2, A1)
        A2 = self.activation_func(Z2)
        
        Z3 = np.dot(W3, A2)
        A3 = self.output_func(Z3)
        
        # Return index of maximum value
        return np.argmax(A3)
        



    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()
        # Initialize history tracking
        self.history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'epochs': []
        }

        for iteration in range(self.epochs):
            # Calculate learning rate for this epoch if using cosine annealing
            if cosine_annealing_lr:
                learning_rate = cosine_annealing(self.learning_rate, 
                                                 iteration, 
                                                 self.epochs, 
                                                 min_lr=0.0)
            else: 
                learning_rate = self.learning_rate
            
            for x, y in zip(x_train, y_train):
                
                # Store current input for backward pass
                self.x_current = x
                
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)
             # Track accuracy history
            train_acc = self.compute_accuracy(x_train, y_train)
            val_acc = self.compute_accuracy(x_val, y_val)
            
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['epochs'].append(iteration + 1)

            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)