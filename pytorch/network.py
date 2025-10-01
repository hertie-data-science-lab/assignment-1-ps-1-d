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
        Forward propagation algorithm.
        Returns the output of the network.
        '''
        # Layer 1: input -> hidden1 with sigmoid activation
        x = self.linear1(x_train)
        x = self.activation_func(x)
        
        # Layer 2: hidden1 -> hidden2 with sigmoid activation
        x = self.linear2(x)
        x = self.activation_func(x)
        
        # Layer 3: hidden2 -> output (logits, no activation)
        x = self.linear3(x)
        
        return x


    def _backward_pass(self, y_train, output):
        '''
        Backpropagation algorithm responsible for updating the weights of the neural network.
        '''
        # Convert y_train to float if needed
        y_train = y_train.float()
        
        # Compute loss
        loss = self.loss_func(output, y_train)
        
        # Backpropagate
        loss.backward()


    def _update_weights(self):
        '''
        Updating the network weights according to stochastic gradient descent.
        '''
        self.optimizer.step()


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
        Prediction making of the network.
        Returns the index of the most likeliest output class
        '''
        # Set to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Flatten input
            x = self._flatten(x)
            
            # Forward pass
            output = self._forward_pass(x)
            
            # Get predicted class (index of max logit)
            predictions = torch.argmax(output, dim=1)
        
        # Set back to training mode
        self.train()
        
        return predictions


    def fit(self, train_loader, val_loader):
        start_time = time.time()
        # Initialize history tracking
        self.history = {"epochs": [], "train_accuracy": [], "val_accuracy": [], "learning_rate": []}

        for iteration in range(self.epochs):
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
                self.optimizer.zero_grad()


                output = self._forward_pass(x)
                self._backward_pass(y, output)
                self._update_weights()
                
            # Calculate and store accuracies for plotting
            train_acc = self.compute_accuracy(train_loader)
            val_acc = self.compute_accuracy(val_loader)
            
            self.history['train_accuracy'].append(train_acc.item())
            self.history['val_accuracy'].append(val_acc.item())
            self.history['epochs'].append(iteration + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            

            self._print_learning_progress(start_time, iteration, train_loader, val_loader)




    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)
