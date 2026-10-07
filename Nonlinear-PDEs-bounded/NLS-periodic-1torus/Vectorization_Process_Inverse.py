import numpy as np

def Vectorization_Process_Inverse(vec_reduced, A_next, torus_idx):
    """
    2026/05/14  Mingwei Fu
    This function reverses the vectorization process:
    It restores a reduced vector back to a tensor (matrix) by
    re-inserting 0s at the resonance positions and reshaping.

    Input:
    vec_reduced: (L_next^{1+b} - 2b) x 1 vector, which is reduced
    A_next:      current truncation scope A_0*A^{r}
    torus_idx:   index of the chosen torus

    Output:
    matrix_ori: L_next x L_next matrix, with 0s re-inserted at
                the resonance positions in [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    """
    
    L_next = int(2 * A_next + 1)
    S_term_p = np.array([torus_idx, 1])
    S_term_m = np.array([-torus_idx, 1])




    # --- Step 1: Create a logical mask in matrix form ---
    mask_matrix = np.ones((L_next, L_next), dtype=bool)

    row_idx_p = int(S_term_p[0] + A_next)
    col_idx_p = int(S_term_p[1] + A_next)
    mask_matrix[row_idx_p, col_idx_p] = False

    row_idx_m = int(S_term_m[0] + A_next)
    col_idx_m = int(S_term_m[1] + A_next)
    mask_matrix[row_idx_m, col_idx_m] = False




    # --- Step 2: Rearrange the mask matrix into a vector ---
    mask_tot = mask_matrix.flatten()




    # --- Step 3: Global zero-padding and filling ---
    vec_full = np.zeros(L_next**2, dtype=vec_reduced.dtype)
    
    # Use the mask to fill the reduced vector back into non-resonant positions
    vec_full[mask_tot] = vec_reduced




    # --- Step 4: Restore to the tensor structure ---
    matrix_ori = vec_full.reshape((L_next, L_next))

    return matrix_ori