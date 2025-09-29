import time

import torch
import torch.nn as nn
import torch.optim as optim


class TorchNetwork(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        self.linear1 = nn.Linear(sizes[0], sizes[1])
        self.linear2 = nn.Linear(sizes[1], sizes[2])
        self.linear3 = nn.Linear(sizes[2], sizes[3])

        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)



    def _forward_pass(self, x_train):
        '''
        Forward pass using PyTorch's automatic operations.
        
        Notice how much simpler this is compared to the scratch implementation!
        PyTorch handles all the matrix operations and gradient tracking automatically.
        
        Args:
            x_train: Input tensor
            
        Returns:
            Raw output logits (before softmax)
        '''
        # Layer 1: Input to first hidden layer
        # self.linear1 automatically does: W1 × x + bias
        h1 = self.activation_func(self.linear1(x_train))  # Apply sigmoid activation
        
        # Layer 2: First hidden to second hidden layer
        h2 = self.activation_func(self.linear2(h1))  # Apply sigmoid activation
        
        # Layer 3: Second hidden to output layer
        # No activation here because our loss function (BCEWithLogitsLoss) 
        # expects raw logits and applies sigmoid/softmax internally
        output = self.linear3(h2)  # Raw output scores
        
        return output


    def _backward_pass(self, y_train, output):
        '''
        Backpropagation using PyTorch's automatic differentiation.
        
        This is the magic of PyTorch! Instead of manually computing gradients
        like we did in the scratch implementation, PyTorch automatically
        tracks all operations and computes gradients for us.
        
        Args:
            y_train: True labels (one-hot encoded)
            output: Network predictions
        '''
        # Convert labels to float type for loss calculation
        y_train = y_train.float()
        
        # Calculate the loss (how wrong our predictions are)
        # BCEWithLogitsLoss combines sigmoid + binary cross entropy
        loss = self.loss_func(output, y_train)
        
        # THE MAGIC: PyTorch automatically computes all gradients!
        # This single line does what took us ~30 lines in the scratch implementation
        # It uses the computational graph to apply the chain rule everywhere
        loss.backward()  # Compute gradients for ALL parameters automatically


    def _update_weights(self):
        '''
        Update weights using PyTorch's built-in optimizer.
        
        Again, PyTorch makes this incredibly simple! The optimizer
        automatically applies the SGD update rule to all parameters.
        '''
        # Apply SGD update to all network parameters
        # This automatically does: W = W - learning_rate * gradient
        # for every single weight and bias in the network
        self.optimizer.step()  # One line replaces our manual weight updates!


    def _flatten(self, x):
        return x.view(x.size(0), -1)       


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def predict(self, x):
        '''
        Make predictions using the trained PyTorch network.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            Predicted class indices (0-9 for digits)
        '''
        # Disable gradient computation for faster inference and less memory usage
        # We don't need gradients when just making predictions
        with torch.no_grad():
            x = self._flatten(x)  # Flatten images to vectors
            output = self._forward_pass(x)  # Get raw logits
            
            # Find the class with highest score for each input
            # dim=1 means we take argmax along the class dimension
            return torch.argmax(output, dim=1)  # Return predicted class indices


    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
                self.optimizer.zero_grad()


                output = self._forward_pass(x)
                self._backward_pass(y, output)
                self._update_weights()

            self._print_learning_progress(start_time, iteration, train_loader, val_loader)




    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)
