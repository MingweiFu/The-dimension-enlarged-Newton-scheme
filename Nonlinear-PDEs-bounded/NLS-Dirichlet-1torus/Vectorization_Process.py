import numpy as np

def Vectorization_Process(matrix_ori, A_next, torus_idx):
    """
    2026/05/14  Mingwei Fu
    This function process rearrangement and deleting of the
    P-equations.

    Input:
    matrix_ori:        L_next x L_next matrix, which is P-equations
                       in [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    A_next:            current truncation scope A_0*A^{r}
    torus_idx:         indices of these chosen torus

    Output:
    matrix_nan:        P-equations values in forms of matrix of size
                       (Ar*A, Ar*A) setting nan at the Q-eqns
    vec_total_nan:     the whole vectors with nan at Q-eqns
    vec_reduced_total: the whole vectors after deleting Q-equation
                       terms
    """
    
    S_term_p = np.array([torus_idx, 1])
    S_term_m = np.array([-torus_idx, 1])




    # Step 1: Set the resonance terms in the P-equation matrix to NaN
    matrix_nan = matrix_ori.copy()

    row_p = int(S_term_p[0] + A_next)
    col_p = int(S_term_p[1] + A_next)
    matrix_nan[row_p, col_p] = np.nan

    row_m = int(S_term_m[0] + A_next)
    col_m = int(S_term_m[1] + A_next)
    matrix_nan[row_m, col_m] = np.nan




    # Step 2: Integrate into a complete global vector containing Q-equation terms
    vec_total_nan = matrix_nan.flatten()




    # Step 3: Obtain the reduced global vector by removing Q-equation terms
    vec_reduced_total = vec_total_nan[~np.isnan(vec_total_nan)]

    return matrix_nan, vec_total_nan, vec_reduced_total