import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        Initialize the class inheriting from scratch.network.Network.
        The method should check whether the residual network is properly initialized.
        '''
        # Check if residual connection is valid
        # For residual connection, hidden layer 1 and hidden layer 2 must have same size
        if sizes[1] != sizes[2]:
            raise ValueError(
                f"Residual connection requires hidden layers to have the same size. "
                f"Got sizes[1]={sizes[1]} and sizes[2]={sizes[2]}. "
                f"Please ensure sizes[1] == sizes[2] for residual connections."
            )
        
        # Call parent constructor to initialize weights and other attributes
        super().__init__(sizes, epochs, learning_rate, random_state)
        
        


    def _forward_pass(self, x_train):
        '''
        Forward propagation algorithm.
        Return the output of the network.
        '''
        # Store parameters for easier access
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3 = self.params['W3']
        
        # Layer 1: input -> hidden1
        self.Z1 = np.dot(W1, x_train)
        self.A1 = self.activation_func(self.Z1)
        
        # Layer 2: hidden1 -> hidden2 WITH RESIDUAL CONNECTION
        # Instead of A2 = activation(Z2), we do A2 = activation(Z2) + A1
        self.Z2 = np.dot(W2, self.A1)
        self.A2 = self.activation_func(self.Z2) + 0.1 * self.A1  # Scale down residual

        # Layer 3: hidden2 -> output
        self.Z3 = np.dot(W3, self.A2)
        self.A3 = self.output_func(self.Z3)
        
        return self.A3
    
    



    def _backward_pass(self, y_train, output):
        '''
        Backpropagation algorithm responsible for updating the weights of the neural network.
        Args:
        Return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
       

        '''
        # Store current input
        x_train = self.x_current
        
        # Output layer gradient (same as standard network)
        dZ3 = output - y_train
        dW3 = np.outer(dZ3, self.A2)
        
        # Hidden layer 2 gradient (accounting for residual connection)
        dA2 = np.dot(self.params['W3'].T, dZ3)
        
        # Since A2 = activation(Z2) + A1, the gradient flows through both paths
        # For the weighted path: d/dZ2 of activation(Z2)
        dZ2 = dA2 * self.activation_func_deriv(self.Z2)
        dW2 = np.outer(dZ2, self.A1)
        
        # Hidden layer 1 gradient (receives gradients from TWO sources)
        # 1. Through W2 (weighted connection): W2.T @ dZ2
        dA1_from_W2 = np.dot(self.params['W2'].T, dZ2)
        
        # 2. Through residual connection (direct skip): dA2
        # Since A2 = activation(Z2) + A1, we have dL/dA1 includes dL/dA2
        dA1_from_residual = dA2
        
        # Combine both gradient contributions
        dA1 = dA1_from_W2 + dA1_from_residual
        
        # Continue backprop to get dW1
        dZ1 = dA1 * self.activation_func_deriv(self.Z1)
        dW1 = np.outer(dZ1, x_train)
        
        return {
            'W1': dW1,
            'W2': dW2,
            'W3': dW3
        }


