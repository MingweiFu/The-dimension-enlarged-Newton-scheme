import numpy as np
from scipy.signal import convolve

def T_construct_opt(u_appro, mu_appro, mu, eps, a, torus_idx, A):
    """
    2026/05/29  Mingwei Fu
    This function constructs the linearized operator 
    T_{A_0*A^{r}}, which is size ( (2A_0*A^{r}+1)^{1+b} ) - b times
    ( (2A_0*A^{r}+1)^{1+b} ) - b.
    MEMORY OPTIMIZED VERSION 

    Input:
    u_appro:   tensor of u(n,k1,k2) coefficients
    mu_appro:  approximate frequencies
    mu:        initial frequency function @(n) mu(n) = |n|^2 + m
    eps:       perturbation quantity
    a:         initial coefficients
    torus_idx: indices of these chosen tori 
    A:         truncation parameter

    Output:
    T: linearized matrix of size ( (2A_0*A^{r}+1)^{1+b} ) - b times
       ( (2A_0*A^{r}+1)^{1+b} ) - b, with resonant entries reduced
    """

    # settings
    Lr     = u_appro.shape[0]     # Lr = 2 * A_0 * A^{r-1} + 1
    Ar     = int((Lr - 1) // 2)   # Ar = A_0 * A^{r-1} 
    A_next = int(Ar * A)          # A_next = A_0 * A^{r}
    L_next = int(2 * A_next + 1)  # L_next = 2 * A_0 * A^{r} + 1
    L_cube = int(L_next**3)       # Total number of elements in the tensor


    # calculate the Q-eqns
    u_appro_bar = np.flip(u_appro.copy())


    # Pre-compute Convolutions
    S1_conv = 2 * convolve(u_appro, u_appro_bar, mode="full")
    S2_conv = convolve(u_appro, u_appro, mode="full")
    Ar_expand = int(2 * Ar)
    Lr_expand = int(2 * Ar_expand + 1)


    # Expand each convolution tensor to the interval [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    offset_2 = int(A_next - Ar_expand)
    S1_next = np.zeros((L_next, L_next, L_next))
    S2_next = np.zeros((L_next, L_next, L_next))
    S1_next[offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand] = S1_conv
    S2_next[offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand] = S2_conv


    # Prepare the 1D vectors for frequency coordinates
    ks = np.arange(-A_next, A_next + 1)
    N_grid, K1_grid, K2_grid = np.meshgrid(ks, ks, ks, indexing="ij")

    N_v = N_grid.flatten()
    K1_v = K1_grid.flatten()
    K2_v = K2_grid.flatten()
    center = int(A_next)


    # Locate the indices of resonance points for single-excision (Strictly Periodic B.C.)
    e_vec1 = np.array([torus_idx[0], 1, 0])
    e_vec2 = np.array([torus_idx[1], 0, 1])

    coords = np.column_stack((N_v, K1_v, K2_v))
    idx_e1 = np.where((coords == e_vec1).all(axis=1))[0][0]
    idx_e2 = np.where((coords == e_vec2).all(axis=1))[0][0]
    idx_e  = [idx_e1, idx_e2]


    # =========================================================================
    # In-place assembly (Memory optimization core)
    # We allocate only one large matrix to act as the final T matrix.
    # =========================================================================
    T_full = np.zeros((L_cube, L_cube))




    # --- Step 1: Column-by-Column construction of Nonlinear part (eps * S) ---
    for col in range(L_cube):
        nc = N_v[col]
        kc1 = K1_v[col]
        kc2 = K2_v[col]

        # S1 part
        dn = N_v - nc
        dk1 = K1_v - kc1
        dk2 = K2_v - kc2
        mask_d = ((np.abs(dn) <= Ar_expand) & (np.abs(dk1) <= Ar_expand) & (np.abs(dk2) <= Ar_expand))
        if np.any(mask_d):
            T_full[mask_d, col] = (eps * S1_next[dn[mask_d] + center, dk1[mask_d] + center, dk2[mask_d] + center])

        # S2 part
        sn = N_v + nc
        sk1 = K1_v + kc1
        sk2 = K2_v + kc2
        mask_s = ((np.abs(sn) <= Ar_expand) & (np.abs(sk1) <= Ar_expand) & (np.abs(sk2) <= Ar_expand))
        if np.any(mask_s):
            T_full[mask_s, col] += (eps * S2_next[sn[mask_s] + center, sk1[mask_s] + center, sk2[mask_s] + center])

    # At this point, T_full is exactly the matrix (eps * S)




    # --- Step 2: Add frequency derivative part B in-place ---
    offset_1 = int(A_next - Ar)
    u_appro_expand = np.zeros((L_next, L_next, L_next))
    u_appro_expand[offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr] = u_appro
    u_curr_v = u_appro_expand.flatten()

    V_row_1 = -K1_v * u_curr_v
    V_row_2 = -K2_v * u_curr_v
    V_col_eps_1 = T_full[idx_e1, :].copy()
    V_col_eps_2 = T_full[idx_e2, :].copy()


    # Add B to T_full column by column to avoid creating an L_cube x L_cube outer product matrix
    for col in range(L_cube):
        T_full[:, col] += ((1 / a[0]) * V_row_1 * V_col_eps_1[col] + (1 / a[1]) * V_row_2 * V_col_eps_2[col])




    # --- Step 3: Add diagonal part D directly to the diagonal of T_full ---
    diag_vec = -(mu_appro[0] * K1_v + mu_appro[1] * K2_v) + mu(N_v)
    np.fill_diagonal(T_full, T_full.diagonal() + diag_vec)




    # --- Step 4: Perform excision on the finalized matrix ---
    T_full = np.delete(T_full, idx_e, axis=0)
    T_full = np.delete(T_full, idx_e, axis=1)

    T = T_full

    return T