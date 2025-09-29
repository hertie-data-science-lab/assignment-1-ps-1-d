"""
Utility functions for neural network operations.

This file contains activation functions, loss functions, and their derivatives
needed for forward propagation and backpropagation.
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function: squashes values to range (0, 1).
    
    Used in hidden layers to introduce non-linearity. Without activation functions,
    multiple layers would just be equivalent to a single linear transformation.
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    """
    Derivative of sigmoid function - needed for backpropagation.
    
    This tells us how much the sigmoid output changes when the input changes slightly.
    We multiply this by the error to get the gradient for weight updates.
    """
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)


def softmax(x):
    """
    Softmax activation function: converts logits to probabilities.
    
    Used in the output layer for classification. Takes raw scores and converts
    them to probabilities that sum to 1. The highest score gets the highest probability.
    
    We subtract x.max() for numerical stability (prevents overflow).
    """
    exps = np.exp(x - x.max())  # Subtract max for numerical stability
    return exps / np.sum(exps, axis=0)  # Normalize to get probabilities


def softmax_deriv(x):
    """
    Derivative of softmax function - needed for backpropagation.
    
    Softmax derivative is more complex because each output depends on all inputs.
    """
    exps = np.exp(x - x.max())
    softmax_output = exps / np.sum(exps, axis=0)
    return softmax_output * (1 - softmax_output)


def mse(y_true, y_pred):
    """
    Mean Squared Error loss function.
    
    Measures how far our predictions are from the true values.
    We square the differences so positive and negative errors don't cancel out.
    
    Formula: MSE = mean((y_true - y_pred)^2)
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_deriv(y_true, y_pred):
    """
    Derivative of MSE loss - tells us direction to adjust predictions.
    
    This is used as the starting point for backpropagation.
    Positive value means prediction was too high, negative means too low.
    """
    return 2 * (y_pred - y_true) / y_pred.shape[0]
