import numpy as np

def Vectorization_Process(tensor_ori, A_next, torus_idx):
    """
    2026/05/25  Mingwei Fu
    This function processes the rearrangement and deletion of the
    P-equations. 

    Input:
    tensor_ori: L_next x L_next x L_next tensor, which contains P-equations
                in [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    A_next:     current truncation scope A_0*A^{r}
    torus_idx:  indices of the chosen basic tori [n1, n2]

    Output:
    tensor_nan:        P-equations values in the form of a tensor of size
                       (L_next, L_next, L_next) with NaN set at the Q-eqns positions
    vec_total_nan:     the unified flat vector containing NaN at Q-eqns positions
    vec_reduced_total: the reduced flat vector after deleting Q-equation terms
    """
    
    S_term_p = np.array([
        [torus_idx[0], 1, 0],  # ( n1, e1)
        [torus_idx[1], 0, 1]   # ( n2, e2)
    ])
      
    S_term_m = np.array([
        [-torus_idx[0], 1, 0], # (-n1, e1)
        [-torus_idx[1], 0, 1]  # (-n2, e2)
    ])


    
    
    # Step 1: Set the resonance terms in the P-equation tensor to NaN
    tensor_nan = tensor_ori.copy()
    
    for j in range(2):
        n_p   = int(S_term_p[j, 0] + A_next)
        k1_p  = int(S_term_p[j, 1] + A_next)
        k2_p  = int(S_term_p[j, 2] + A_next)
        tensor_nan[n_p, k1_p, k2_p] = np.nan
        
        n_m   = int(S_term_m[j, 0] + A_next)
        k1_m  = int(S_term_m[j, 1] + A_next)
        k2_m  = int(S_term_m[j, 2] + A_next)
        tensor_nan[n_m, k1_m, k2_m] = np.nan




    # Step 2: Integrate into a complete global vector containing Q-equation terms
    vec_total_nan = tensor_nan.flatten()

    
    
    
    
    # Step 3: Obtain the reduced global vector by removing Q-equation terms
    vec_reduced_total = vec_total_nan[~np.isnan(vec_total_nan)]

    return tensor_nan, vec_total_nan, vec_reduced_total