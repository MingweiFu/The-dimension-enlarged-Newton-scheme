import numpy as np
from scipy.signal import convolve

def T_construct_opt(u_appro, mu_appro, mu, eps, a, torus_idx, A):
    """
    2026/10/09  Mingwei Fu
    This function constructs the linearized operator 
    T_{A_0*A^{r}}, which is size (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b times
    (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b.
    MEMORY OPTIMIZED VERSION

    Input:
    u_appro:   tensor of u(n,k1,k2) coefficients
    mu_appro:  approximate frequencies
    mu:        basic frequency function @(n) mu(n) = sqrt(n^2 + n^4)
    eps:       perturbation quantity
    a:         initial coefficients
    torus_idx: indices of these chosen tori 
    A:         truncation parameter

    Output:
    T: linearized matrix of size (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b times
       (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b, with the Q-equation positions 
       and all u(0,k) positions removed
    """

    # settings
    Lr     = u_appro.shape[0]     # Lr = 2 * A_0 * A^{r-1} + 1
    Ar     = int((Lr - 1) // 2)   # Ar = A_0 * A^{r-1} 
    A_next = int(Ar * A)          # A_next = A_0 * A^{r}
    L_next = int(2 * A_next + 1)  # L_next = 2 * A_0 * A^{r} + 1
    L_cube = int(L_next**3)       # Total number of elements in the tensor


    # Construct S
    S_conv = 2 * u_appro
    Ar_expand = int(Ar)
    Lr_expand = int(2 * Ar_expand + 1)


    # Expand each convolution tensor to the interval [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    offset_2 = int(A_next - Ar_expand)
    S_next = np.zeros((L_next, L_next, L_next))
    
    S_next[offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand] = S_conv


    # Prepare the 1D vectors for frequency coordinates
    ks = np.arange(-A_next, A_next + 1)
    N_grid, K1_grid, K2_grid = np.meshgrid(ks, ks, ks, indexing="ij")

    N_v = N_grid.flatten()
    K1_v = K1_grid.flatten()
    K2_v = K2_grid.flatten()
    center = int(A_next)


    # Locate the indices of resonance points for dual-excision
    e_vec1_p = np.array([ torus_idx[0],  1,  0])
    e_vec1_m = np.array([-torus_idx[0], -1,  0])
    e_vec2_p = np.array([ torus_idx[1],  0,  1])
    e_vec2_m = np.array([-torus_idx[1],  0, -1])

    coords = np.column_stack((N_v, K1_v, K2_v))
    idx_e1_p = np.where((coords == e_vec1_p).all(axis=1))[0][0]
    idx_e1_m = np.where((coords == e_vec1_m).all(axis=1))[0][0]
    idx_e2_p = np.where((coords == e_vec2_p).all(axis=1))[0][0]
    idx_e2_m = np.where((coords == e_vec2_m).all(axis=1))[0][0]
    
    # Locate all u(0,k1,k2) positions
    idx_n_zero = np.where(N_v == 0)[0]
    
    # Indices deleted from both rows and columns
    idx_drop = np.unique(np.concatenate((idx_n_zero, np.array([idx_e1_p, idx_e1_m, idx_e2_p, idx_e2_m]))))




    # =========================================================================
    # In-place assembly (Memory optimization core)
    # We allocate only one large matrix to act as the final T matrix.
    # =========================================================================
    T_full = np.zeros((L_cube, L_cube))




    # --- Step 1: Column-by-Column construction of Nonlinear part (eps * W * S) ---
    for col in range(L_cube):
        nc = N_v[col]
        kc1 = K1_v[col]
        kc2 = K2_v[col]

        dn = N_v - nc
        dk1 = K1_v - kc1
        dk2 = K2_v - kc2
        mask_d = ((np.abs(dn) <= Ar_expand) & (np.abs(dk1) <= Ar_expand) & (np.abs(dk2) <= Ar_expand))
        if np.any(mask_d):
            T_full[mask_d, col] = eps * N_v[mask_d]**2 * S_next[dn[mask_d] + center, dk1[mask_d] + center, dk2[mask_d] + center]

    # At this point, T_full is exactly the matrix (eps * W * S)




    # --- Step 2: Add Frequency Derivative part B in-place ---
    offset_1 = int(A_next - Ar)
    u_appro_expand = np.zeros((L_next, L_next, L_next))
    u_appro_expand[offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr] = u_appro
    u_curr_v = u_appro_expand.flatten()
    
    # <k, omega>
    omega_dot_k_v = ( mu_appro[0] * K1_v + mu_appro[1] * K2_v )

    V_row_1 = np.zeros(L_cube)
    V_row_2 = np.zeros(L_cube)
    
    nonzero_n_vec_mask = (N_v != 0)

    V_row_1[nonzero_n_vec_mask] = (omega_dot_k_v[nonzero_n_vec_mask] * K1_v[nonzero_n_vec_mask]) * u_curr_v[nonzero_n_vec_mask]
    V_row_2[nonzero_n_vec_mask] = (omega_dot_k_v[nonzero_n_vec_mask] * K2_v[nonzero_n_vec_mask]) * u_curr_v[nonzero_n_vec_mask]

    V_col_eps_1 = T_full[idx_e1_p, :].copy()
    V_col_eps_2 = T_full[idx_e2_p, :].copy()


    # Add B to T_full column by column to avoid creating an L_cube x L_cube outer product matrix
    for col in range(L_cube):
        T_full[:, col] += ( ( -1 / (a[0] * mu_appro[0]) ) * V_row_1 * V_col_eps_1[col] 
                          + ( -1 / (a[1] * mu_appro[1]) ) * V_row_2 * V_col_eps_2[col] )



    # --- Step 3: Add Diagonal part D directly to the diagonal of T_full ---
    diag_vec = np.zeros(L_cube)
    diag_vec[nonzero_n_vec_mask] = mu(N_v[nonzero_n_vec_mask])**2 - (mu_appro[0] * K1_v[nonzero_n_vec_mask] + mu_appro[1] * K2_v[nonzero_n_vec_mask])**2
    np.fill_diagonal(T_full, T_full.diagonal() + diag_vec)




    # --- Step 4: Perform Dual Excision on the finalized matrix ---
    keep_mask = np.ones(L_next**3, dtype=bool)
    keep_mask[idx_drop] = False
    keep_idx = np.flatnonzero(keep_mask)

    T = T_full[np.ix_(keep_idx, keep_idx)]

    return T