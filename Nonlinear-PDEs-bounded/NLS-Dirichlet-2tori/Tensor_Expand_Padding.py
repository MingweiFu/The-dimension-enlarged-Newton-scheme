import numpy as np

def Tensor_Expand_Padding(T_small, A, B):
    """ 
    2026/05/25  Mingwei Fu
    This function expands a Fourier coefficient tensor from a smaller
    scope [-A, A]^3 to a larger scope [-B, B]^3 using zero-padding.
    The center of the matrix is maintained at (0,0).

    Input:
    T_small:  The original tensor of size (2*A + 1) x (2*A + 1) x (2*A + 1)
    A:        The original truncation radius
    B:        The target truncation radius (B > A)

    Output:
    T_large:  The expanded tensor of size (2*B + 1) x (2*B + 1) x (2*B + 1)
              with T_small at its center and zero-padding elsewhere.
    """
    
    if B <= A:
        raise ValueError('B must be greater than A')
        
    if T_small.shape[0] != int(2 * A + 1):
        raise ValueError('T_small dimension does not match parameter A')
        
    L_A = int(2 * A + 1)
    L_B = int(2 * B + 1)
    
    T_large = np.zeros((L_B, L_B, L_B))
    
    offset = int(B - A)
    
    T_large[offset : offset + L_A, offset : offset + L_A, offset : offset + L_A] = T_small
    
    return T_large
     