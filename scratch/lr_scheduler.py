import numpy as np

def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0.0):
    """
    The scheduler decreases the learning rate according to the formula:
    ℓt = ℓT + (ℓ0 - ℓT)/2 * (1 + cos(πt/T))
    
    where:
    - ℓ0 is the initial learning rate (initial_lr)
    - ℓT is the final learning rate after T iterations (min_lr)
    - t is the current epoch
    - T is the total number of epochs (total_epochs)
    
    Args:
        initial_lr (float): Initial learning rate (ℓ0).
        epoch (int): Current epoch number (t).
        total_epochs (int): Total number of epochs (T).
        min_lr (float): Minimum learning rate to reach (ℓT).
        
    Returns:
        Adjusted learning rate for the current epoch (ℓt).
    """
    if total_epochs == 0:
        return initial_lr
    
    # Calculate the cosine annealing learning rate
    # ℓt = ℓT + (ℓ0 - ℓT)/2 * (1 + cos(πt/T))
    cos_inner = np.pi * epoch / total_epochs
    cos_out = np.cos(cos_inner)
    lr = min_lr + (initial_lr - min_lr) / 2.0 * (1.0 + cos_out)
    
    return lr