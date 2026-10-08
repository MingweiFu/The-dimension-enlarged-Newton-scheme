import numpy as np

def Vectorization_Process_Inverse(vec_reduced, A_next, torus_idx):
    """
    2026/06/30  Mingwei Fu
    This function reverses the vectorization process:
    It restores a reduced vector back to a tensor by
    re-inserting 0s at the resonance positions and reshaping.

    Input:
    vec_reduced: (L_next^{1+b} - 4b) x 1 vector, which is reduced
    A_next:      current truncation scope A_0*A^{r}
    torus_idx:   indices of the chosen tori

    Output:
    tensor_ori:  L_next x L_next x L_next tensor, with 0s re-inserted
                 at the resonance positions in [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    """

    L_next = int(2 * A_next + 1)
    S_term = np.array([
        [ torus_idx[0],  1,  0],  # ( n1,  e1)
        [-torus_idx[0],  1,  0],  # (-n1,  e1)
        [ torus_idx[0], -1,  0],  # ( n1, -e1)
        [-torus_idx[0], -1,  0],  # (-n1, -e1)

        [ torus_idx[1],  0,  1],  # ( n2,  e2)
        [-torus_idx[1],  0,  1],  # (-n2,  e2)
        [ torus_idx[1],  0, -1],  # ( n2, -e2)
        [-torus_idx[1],  0, -1]   # (-n2, -e2)
    ])




    # --- Step 1: Create a logical mask in tensor form ---
    mask_tensor = np.ones((L_next, L_next, L_next), dtype=bool)

    for j in range(8):
        n = int(S_term[j, 0] + A_next)
        k1 = int(S_term[j, 1] + A_next)
        k2 = int(S_term[j, 2] + A_next)
        mask_tensor[n, k1, k2] = False




    # --- Step 2: Rearrange the mask matrix into a vector ---
    mask_tot = mask_tensor.flatten()




    # --- Step 3: Global zero-padding and filling ---
    vec_full = np.zeros(L_next**3, dtype=vec_reduced.dtype)

    # Use the mask to fill the reduced vector back into non-resonant positions
    vec_full[mask_tot] = vec_reduced




    # --- Step 4: Restore to the tensor structure ---
    tensor_ori = vec_full.reshape((L_next, L_next, L_next))

    return tensor_ori