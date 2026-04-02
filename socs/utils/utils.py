import copy
import numpy as np

def row_normalize(A):
    """
    Computes a row-normalized copy of the input array A

    Parameters
    ----------
    A: np.ndarray
        Array to be row-normalized
    
    Returns
    -------
    A_n: np.ndarray
        
    """
    A_n = copy.deepcopy(A)
    for x in range(A.shape[0]):
        A_n[x,:] = A_n[x,:]/np.sum(A_n[x,:])
    return A_n