import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        Initialize ResNet (Residual Network) with skip connections.
        
        ResNets add "skip connections" that allow information to flow directly
        from one layer to a later layer. This helps with training deeper networks
        and can improve gradient flow during backpropagation.
        
        Args:
            sizes: Network architecture [input, hidden1, hidden2, output]
            epochs: Number of training epochs
            learning_rate: Learning rate for SGD
            random_state: Random seed for reproducible results
        '''
        # IMPORTANT: For skip connections to work mathematically,
        # the layers being connected must have the same number of neurons
        # (so we can add them together: layer2_output + layer1_output)
        
        if len(sizes) == 4 and sizes[1] != sizes[2]:
            # If hidden layers have different sizes, make them the same
            # Use the first hidden layer size for both
            modified_sizes = [sizes[0], sizes[1], sizes[1], sizes[3]]  # e.g., 784→128→128→10
            print(f"Modified architecture for ResNet: {sizes} → {modified_sizes}")
            print("Reason: Skip connections require matching layer dimensions")
            sizes = modified_sizes
        
        # Initialize the parent Network class with our (possibly modified) architecture
        super(ResNetwork, self).__init__(sizes, epochs, learning_rate, random_state)
        
        


    def _forward_pass(self, x_train):
        '''
        Forward pass with residual (skip) connections.
        
        The key difference from regular networks: instead of just passing
        information through layers sequentially, we also add a "shortcut"
        that allows information to skip a layer and be added later.
        
        Regular: input → layer1 → layer2 → output
        ResNet:  input → layer1 → layer2 + layer1 → output
                               ↑_____shortcut____↑
        
        Returns:
            Network output with residual connections applied
        '''
        # Store input for backpropagation (same as regular network)
        self.x_input = x_train
        
        # Layer 1: Transform input to first hidden layer (standard)
        z1 = np.dot(self.params['W1'], x_train)  # Linear transformation
        a1 = self.activation_func(z1)  # Apply sigmoid activation
        
        # Layer 2: This is where ResNet magic happens!
        # Instead of just: layer2 = activation(W2 × layer1)
        # We do: layer2 = activation(W2 × layer1) + layer1 (residual connection)
        z2 = np.dot(self.params['W2'], a1)  # Standard linear transformation
        a2_raw = self.activation_func(z2)  # Standard activation
        
        # THE RESIDUAL CONNECTION: Add the input of this layer to its output
        # This creates a "highway" for information to flow directly through
        a2 = a2_raw + a1  # Skip connection: combine processed and original information
        
        # Layer 3: Transform to output (standard)
        z3 = np.dot(self.params['W3'], a2)  # Use the residual-enhanced layer2
        output = self.output_func(z3)  # Apply softmax for probabilities
        
        # Store intermediate values for backpropagation
        # We need both the raw activation and the residual-enhanced version
        self.z1, self.a1 = z1, a1  # Layer 1 values
        self.z2, self.a2_raw, self.a2 = z2, a2_raw, a2  # Layer 2: raw and with residual
        self.z3 = z3  # Output layer
        
        return output



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        The method should also account for the residual connection in the hidden layer.

        '''
        # Start from output layer - compute error
        output_error = output - y_train  # (10,) - shape matches output
        
        # Layer 3 gradients (output layer)
        # dL/dW3 = output_error * a2^T (a2 includes residual connection)
        dW3 = np.outer(output_error, self.a2)  # (10, 128)
        
        # Propagate error backwards to layer 2 (with residual connection)
        # dL/da2 = W3^T * output_error
        hidden2_error = np.dot(self.params['W3'].T, output_error)  # (128, 10) × (10,) = (128,)
        
        # For residual connection: a2 = a2_raw + a1
        # So dL/da2_raw = dL/da2 (since da2/da2_raw = 1)
        # And dL/da1 gets contributions from both direct path and residual path
        hidden2_raw_error = hidden2_error * self.activation_func_deriv(self.z2)  # Error through activation
        
        # Layer 2 gradients
        # dL/dW2 = hidden2_raw_error * a1^T  
        dW2 = np.outer(hidden2_raw_error, self.a1)  # (128, 128)
        
        # Propagate error backwards to layer 1
        # Error comes from two paths: direct (through W2) and residual (direct pass-through)
        # dL/da1 = W2^T * hidden2_raw_error + hidden2_error (residual path)
        hidden1_error_from_W2 = np.dot(self.params['W2'].T, hidden2_raw_error)  # Through W2
        hidden1_error_from_residual = hidden2_error  # Direct residual path
        hidden1_error = (hidden1_error_from_W2 + hidden1_error_from_residual) * self.activation_func_deriv(self.z1)
        
        # Layer 1 gradients  
        # dL/dW1 = hidden1_error * x^T
        dW1 = np.outer(hidden1_error, self.x_input)  # (128, 784)
        
        return {
            'W1': dW1,
            'W2': dW2, 
            'W3': dW3
        }


