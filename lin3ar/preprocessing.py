import numpy as np

def standard_scale(X):
    """
    Standardize the dataset to have zero mean and unit variance.
    
    Parameters:
        X (ndarray): input data of shape (n_samples, n_features)
    
    Returns:
        ndarray: standardized data
    """

    X = X.to_numpy() if hasattr(X, 'to_numpy') else X
    
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)

    std[std == 0] = 1


    return (X - mean) / std