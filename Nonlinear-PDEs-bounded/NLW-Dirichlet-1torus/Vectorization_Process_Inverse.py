import numpy as np

def Vectorization_Process_Inverse(vec_reduced, A_next, torus_idx):
    """
    2026/06/29  Mingwei Fu
    This function reverses the vectorization process:
    It restores a reduced vector back to a tensor (matrix) by
    re-inserting 0s at the resonance positions and reshaping.

    Input:
    vec_reduced: (L_next^{1+b} - 4b) x 1 vector, which is reduced
    A_next:      current truncation scope A_0*A^{r}
    torus_idx:   index of the chosen torus

    Output:
    matrix_ori: L_next x L_next matrix, with 0s re-inserted at
                the resonance positions in [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    """
    
    L_next = int(2 * A_next + 1)
    S_term_pp = np.array([ torus_idx,  1])
    S_term_mp = np.array([-torus_idx,  1])
    S_term_pm = np.array([ torus_idx, -1])
    S_term_mm = np.array([-torus_idx, -1])




    # --- Step 1: Create a logical mask in matrix form ---
    mask_matrix = np.ones((L_next, L_next), dtype=bool)

    row_idx_pp = int(S_term_pp[0] + A_next)
    col_idx_pp = int(S_term_pp[1] + A_next)
    mask_matrix[row_idx_pp, col_idx_pp] = False

    row_idx_mp = int(S_term_mp[0] + A_next)
    col_idx_mp = int(S_term_mp[1] + A_next)
    mask_matrix[row_idx_mp, col_idx_mp] = False

    row_idx_pm = int(S_term_pm[0] + A_next)
    col_idx_pm = int(S_term_pm[1] + A_next)
    mask_matrix[row_idx_pm, col_idx_pm] = False

    row_idx_mm = int(S_term_mm[0] + A_next)
    col_idx_mm = int(S_term_mm[1] + A_next)
    mask_matrix[row_idx_mm, col_idx_mm] = False




    # --- Step 2: Rearrange the mask matrix into a vector ---
    mask_tot = mask_matrix.flatten()




    # --- Step 3: Global zero-padding and filling ---
    vec_full = np.zeros(L_next**2, dtype=vec_reduced.dtype)
    
    # Use the mask to fill the reduced vector back into non-resonant positions
    vec_full[mask_tot] = vec_reduced




    # --- Step 4: Restore to the tensor structure ---
    matrix_ori = vec_full.reshape((L_next, L_next))

    return matrix_ori