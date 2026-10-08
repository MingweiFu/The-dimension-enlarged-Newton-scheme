import numpy as np

def Matrix_Expand_Padding(M_small, A, B):
    """
    2026/06/29  Mingwei Fu
    This function expands a Fourier coefficient matrix from a smaller
    scope [-A, A]^2 to a larger scope [-B, B]^2 using zero-padding.
    The center of the matrix is maintained at (0,0).

    Input:
    M_small:  The original matrix of size (2*A + 1) x (2*A + 1)
    A:        The original truncation radius
    B:        The target truncation radius (B > A)

    Output:
    M_large:  The expanded matrix of size (2*B + 1) x (2*B + 1)
              with M_small at its center and zero-padding elsewhere.
    """

    if B <= A:
        raise ValueError('B must be greater than A')

    if M_small.shape[0] != int(2 * A + 1):
        raise ValueError('M_small length does not match parameter A')

    L_A = int(2 * A + 1)
    L_B = int(2 * B + 1)

    M_large = np.zeros((L_B, L_B))

    offset = int(B - A)

    M_large[offset : offset + L_A, offset : offset + L_A] = M_small

    return M_large