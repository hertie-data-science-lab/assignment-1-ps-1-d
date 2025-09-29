import numpy as np

def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0.0):
    """
    Cosine annealing learning rate scheduler.
    
    This creates a smooth decrease in learning rate following a cosine curve.
    At the beginning, learning rate decreases slowly, then faster in the middle,
    then slowly again toward the end. This often leads to better final performance.
    
    Args:
        initial_lr (float): Starting learning rate (e.g., 0.1)
        epoch (int): Current training epoch (0, 1, 2, ...)
        total_epochs (int): Total number of training epochs
        min_lr (float): Minimum learning rate to reach (usually 0.0)
        
    Returns:
        float: Adjusted learning rate for the current epoch
    """
    # Cosine annealing formula creates a smooth curve from initial_lr to min_lr
    # Formula: lr = min_lr + (initial_lr - min_lr) * (1 + cos(π * progress)) / 2
    # where progress = epoch / total_epochs goes from 0 to 1
    
    # Calculate how far through training we are (0 to 1)
    progress = epoch / total_epochs
    
    # Use cosine function to create smooth decay curve
    # cos(0) = 1 (start), cos(π) = -1 (end)
    cos_factor = np.cos(np.pi * progress)
    
    # Transform cosine output from [-1, 1] to [0, 1] range
    # Then scale between min_lr and initial_lr
    lr = min_lr + (initial_lr - min_lr) * (1 + cos_factor) / 2
    
    return lr