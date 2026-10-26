import numpy as np
from scipy.signal import convolve

def T_construct_opt(u_appro, mu_appro, mu, eps, a, torus_idx, A):
    """
    2026/10/08  Mingwei Fu
    This function constructs the linearized operator
    T_{A_0*A^{r}}, which is size (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b times
    (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b.
    MEMORY OPTIMIZED VERSION

    Input:
    u_appro:   matrix of u(n,k) coefficients
    mu_appro:  approximate frequencies
    mu:        initial frequency function @(n) mu(n) = n^3 + m*n
    eps:       perturbation quantity
    a:         initial coefficients
    torus_idx: index of this chosen torus
    A:         truncation parameter

    Output:
    T: linearized matrix of size (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b times
       (2A_0*A^{r}+1)^{1+b} - 2b - (2A_0*A^{r}+1)^b, with the Q-equation positions 
       and all u(0,k) positions removed
    """

    # settings 
    Lr     = u_appro.shape[0]     # Lr = 2 * A_0 * A^{r-1} + 1
    Ar     = int((Lr - 1) / 2)    # Ar = A_0 * A^{r-1}
    A_next = int(Ar * A)          # A_next = A_0 * A^{r}
    L_next = int(2 * A_next + 1)  # L_next = 2 * A_0 * A^{r} + 1


    # Construct S
    S_conv = 2 * u_appro
    Ar_expand = int(Ar)
    Lr_expand = int(2 * Ar_expand + 1)


    # Expand S matrix to the interval [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    offset_2 = int(A_next - Ar_expand)
    S_next = np.zeros((L_next, L_next))
    
    S_next[offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand] = S_conv


    # Prepare the 1D vectors for frequency coordinates
    ks = np.arange(-A_next, A_next + 1)
    N_grid, K_grid = np.meshgrid(ks, ks, indexing='ij')
    
    N_v = N_grid.flatten()
    K_v = K_grid.flatten()
    center = int(A_next)


    # Locate the indices of resonance points and u(0,k) positions
    e_vec_p = np.array([torus_idx, 1])
    e_vec_m = np.array([-torus_idx, -1])
    coords = np.column_stack((N_v, K_v))
    idx_e1 = np.where((coords == e_vec_p).all(axis=1))[0][0]
    idx_e2 = np.where((coords == e_vec_m).all(axis=1))[0][0]
    
    # Locate all u(0,k) positions
    idx_n_zero = np.where(N_v == 0)[0]
    
    idx_drop = np.unique(np.concatenate((idx_n_zero, np.array([idx_e1, idx_e2]))))
    
    


    # =========================================================================
    # In-place assembly (Memory optimization core)
    # We allocate only one large matrix to act as the final T matrix.
    # =========================================================================
    T_full = np.zeros((L_next**2, L_next**2))




    # --- Step 1: Column-by-Column construction of Nonlinear part (eps * S) ---
    for col in range(L_next**2):
        nc = N_v[col]
        kc = K_v[col]
        
        dn = N_v - nc  
        dk = K_v - kc
        mask_d = (np.abs(dn) <= Ar_expand) & (np.abs(dk) <= Ar_expand)
        if np.any(mask_d):
            T_full[mask_d, col] = eps * N_v[mask_d] * S_next[dn[mask_d] + center, dk[mask_d] + center]
        
    # At this point, T_full is exactly the matrix (eps * W * S)




    # --- Step 2: Add Frequency Derivative part B in-place ---
    offset_1 = int(A_next - Ar)
    u_appro_expand = np.zeros((L_next, L_next))  
    u_appro_expand[offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr] = u_appro
    
    V_row = np.zeros(L_next**2)
    nonzero_n_vec_mask = (N_v != 0)
    V_row[nonzero_n_vec_mask] = K_v[nonzero_n_vec_mask] * u_appro_expand.flatten()[nonzero_n_vec_mask]
    
    V_col_eps = T_full[idx_e1, :].copy()


    # Add B to T_full column by column
    for col in range(L_next**2):
        T_full[:, col] += (-1 / a) * V_row * V_col_eps[col]




    # --- Step 3: Add Diagonal part D directly to the diagonal of T_full ---
    diag_vec = np.zeros(L_next**2)
    diag_vec[nonzero_n_vec_mask] = mu_appro * K_v[nonzero_n_vec_mask] - mu(N_v[nonzero_n_vec_mask])
    np.fill_diagonal(T_full, T_full.diagonal() + diag_vec)




    # --- Step 4: Perform Excision on the finalized matrix ---
    keep_mask = np.ones(L_next**2, dtype=bool)
    keep_mask[idx_drop] = False
    keep_idx = np.flatnonzero(keep_mask)

    T = T_full[np.ix_(keep_idx, keep_idx)]

    return T