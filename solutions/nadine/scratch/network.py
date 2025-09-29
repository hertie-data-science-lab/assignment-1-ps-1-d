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
        Forward propagation through the neural network.
        
        This method computes the output of the network by passing the input
        through each layer with weights and activation functions.
        
        Args:
            x_train: Input data (784 features for MNIST)
            
        Returns:
            output: Network predictions (10 probabilities for digit classes)
        '''
        # Store input for use in backpropagation later
        self.x_input = x_train
        
        # Layer 1: Transform input to first hidden layer
        # Matrix multiplication: weights × input + apply sigmoid activation
        z1 = np.dot(self.params['W1'], x_train)  # Linear transformation: (128, 784) × (784,) = (128,)
        a1 = self.activation_func(z1)  # Apply sigmoid: squash values to (0,1) range
        
        # Layer 2: Transform first hidden layer to second hidden layer  
        z2 = np.dot(self.params['W2'], a1)  # Linear transformation: (64, 128) × (128,) = (64,)
        a2 = self.activation_func(z2)  # Apply sigmoid activation again
        
        # Layer 3: Transform second hidden layer to output layer
        z3 = np.dot(self.params['W3'], a2)  # Linear transformation: (10, 64) × (64,) = (10,)
        output = self.output_func(z3)  # Apply softmax: convert to probabilities that sum to 1
        
        # Store all intermediate values - we need these for backpropagation
        # z values are pre-activation, a values are post-activation
        self.z1, self.a1 = z1, a1  # First layer values
        self.z2, self.a2 = z2, a2  # Second layer values
        self.z3 = z3  # Output layer pre-activation
        
        return output


    def _backward_pass(self, y_train, output):
        '''
        Backpropagation algorithm to compute gradients.
        
        This is the heart of neural network learning! We compute how much
        each weight contributed to the error and calculate gradients to update them.
        We work backwards from output to input using the chain rule.
        
        Args:
            y_train: True labels (one-hot encoded)
            output: Network predictions from forward pass
            
        Returns:
            Dictionary containing gradients for each weight matrix
        '''
        # STEP 1: Calculate error at output layer
        # How far off were our predictions? (prediction - true_value)
        output_error = output - y_train  # Shape: (10,) - one error per class
        
        # STEP 2: Calculate gradients for output layer weights (W3)
        # Gradient = error × input_to_this_layer
        # We use outer product to get the full gradient matrix
        dW3 = np.outer(output_error, self.a2)  # Shape: (10, 64)
        
        # STEP 3: Propagate error backwards to hidden layer 2
        # How much did each neuron in layer 2 contribute to the output error?
        # We multiply by the transpose of weights to "reverse" the forward pass
        hidden2_error = np.dot(self.params['W3'].T, output_error)  # Shape: (64,)
        
        # Apply the derivative of the activation function
        # This tells us how sensitive the layer output is to small changes
        hidden2_error = hidden2_error * self.activation_func_deriv(self.z2)
        
        # STEP 4: Calculate gradients for hidden layer 2 weights (W2)
        dW2 = np.outer(hidden2_error, self.a1)  # Shape: (64, 128)
        
        # STEP 5: Propagate error backwards to hidden layer 1
        # Same process: multiply by weight transpose to go backwards
        hidden1_error = np.dot(self.params['W2'].T, hidden2_error)  # Shape: (128,)
        
        # Apply activation function derivative again
        hidden1_error = hidden1_error * self.activation_func_deriv(self.z1)
        
        # STEP 6: Calculate gradients for hidden layer 1 weights (W1)
        # Use original input (stored during forward pass)
        dW1 = np.outer(hidden1_error, self.x_input)  # Shape: (128, 784)
        
        # Return all gradients - these tell us how to update each weight
        return {
            'W1': dW1,  # Gradients for input → hidden1 weights
            'W2': dW2,  # Gradients for hidden1 → hidden2 weights
            'W3': dW3   # Gradients for hidden2 → output weights
        }


    def _update_weights(self, weights_gradient, learning_rate):
        '''
        Update network weights using Stochastic Gradient Descent (SGD).
        
        This is where the actual learning happens! We adjust each weight
        in the direction that reduces the error, scaled by the learning rate.
        
        Args:
            weights_gradient: Dictionary of gradients from backpropagation
            learning_rate: How big steps to take (controls learning speed)
        '''
        # SGD update rule: new_weight = old_weight - learning_rate × gradient
        # The gradient points in the direction of steepest increase in error,
        # so we subtract it to move toward lower error
        
        self.params['W1'] -= learning_rate * weights_gradient['W1']  # Update input → hidden1 weights
        self.params['W2'] -= learning_rate * weights_gradient['W2']  # Update hidden1 → hidden2 weights
        self.params['W3'] -= learning_rate * weights_gradient['W3']  # Update hidden2 → output weights


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
        Make a prediction for a single input.
        
        Args:
            x: Input data (784 pixel values for MNIST)
            
        Returns:
            Predicted class (0-9 for digit recognition)
        '''
        # Run input through the network to get probabilities for each class
        output = self._forward_pass(x)  # Shape: (10,) - probability for each digit
        
        # Return the class with the highest probability
        return np.argmax(output)  # Index of maximum value (0-9)



    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()
        
        # Initialize history tracking
        self.history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                
                if cosine_annealing_lr:
                    learning_rate = cosine_annealing(self.learning_rate, 
                                                     iteration, 
                                                     len(x_train), 
                                                     self.learning_rate)
                else: 
                    learning_rate = self.learning_rate
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)

            # Track metrics
            train_acc = self.compute_accuracy(x_train, y_train)
            val_acc = self.compute_accuracy(x_val, y_val)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(learning_rate)
            
            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
