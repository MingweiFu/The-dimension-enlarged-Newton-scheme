import numpy as np

def Vectorization_Process_Inverse(vec_reduced, A_next, torus_idx):
    """
    2026/10/09  Mingwei Fu
    This function reverses the vectorization process:
    It restores a reduced vector back to a tensor by 
    re-inserting 0s at the resonance and u(0, k1, k2) positions and reshaping.

    Input:
    vec_reduced: (L_next^{1+b} - 2b - L_next^b) x 1 vector, which is reduced
    A_next:      current truncation scope A_0*A^{r}
    torus_idx:   indices of the chosen tori

    Output:
    tensor_ori:  L_next x L_next x L_next tensor, with 0s re-inserted
                 at the resonance and u(0, k1, k2) positions in [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    """

    L_next = int(2 * A_next + 1)
    S_term_p = np.array([[torus_idx[0], 1, 0],     # ( n1,  e1)
                         [torus_idx[1], 0, 1]])    # ( n2,  e2)
    
    S_term_m = np.array([[-torus_idx[0], -1, 0],   # (-n1, -e1)
                         [-torus_idx[1], 0, -1]])  # (-n2, -e2)




    # --- Step 1: Create a logical mask in tensor form ---
    mask_tensor = np.ones((L_next, L_next, L_next), dtype=bool)
    
    n_zero_idx = int(A_next)
    mask_tensor[n_zero_idx, :, :] = False

    for j in range(2):
        n_p = int(S_term_p[j, 0] + A_next)
        k1_p = int(S_term_p[j, 1] + A_next)
        k2_p = int(S_term_p[j, 2] + A_next)
        mask_tensor[n_p, k1_p, k2_p] = False

        n_m = int(S_term_m[j, 0] + A_next)
        k1_m = int(S_term_m[j, 1] + A_next)
        k2_m = int(S_term_m[j, 2] + A_next)
        mask_tensor[n_m, k1_m, k2_m] = False




    # --- Step 2: Rearrange the mask tensor into a vector ---
    mask_tot = mask_tensor.flatten()




    # --- Step 3: Global zero-padding and filling ---
    vec_full = np.zeros(L_next**3, dtype=vec_reduced.dtype)

    # Use the mask to fill the reduced vector back into non-resonant positions
    vec_full[mask_tot] = vec_reduced




    # --- Step 4: Restore to the tensor structure ---
    tensor_ori = vec_full.reshape((L_next, L_next, L_next))

    return tensor_ori