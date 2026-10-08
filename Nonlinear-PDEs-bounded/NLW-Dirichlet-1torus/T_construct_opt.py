import numpy as np
from scipy.signal import convolve

def T_construct_opt(u_appro, mu_appro, mu, eps, a, torus_idx, A):
    """
    2026/06/30  Mingwei Fu
    This function constructs the linearized operator
    T_{A_0*A^{r}}, which is size ( (2A_0*A^{r}+1)^{1+b} ) - 4b times
    ( (2A_0*A^{r}+1)^{1+b} ) - 4b.
    MEMORY OPTIMIZED VERSION

    Input:
    u_appro:   matrix of u(n,k) coefficients
    mu_appro:  approximate frequencies
    mu:        initial frequency function @(n) mu(n) = (|n|^2 + rho)^(1/2)
    eps:       perturbation quantity
    a:         initial coefficients
    torus_idx: index of this chosen torus
    A:         truncation parameter

    Output:
    T: linearized matrix of size ( (2A_0*A^{r}+1)^{1+b} ) - 4b times
       ( (2A_0*A^{r}+1)^{1+b} ) - 4b, with resonant entries reduced
    """

    # settings 
    Lr     = u_appro.shape[0]     # Lr = 2 * A_0 * A^{r-1} + 1
    Ar     = (Lr - 1) / 2         # Ar = A_0 * A^{r-1}
    A_next = int(Ar * A)          # A_next = A_0 * A^{r}
    L_next = int(2 * A_next + 1)  # L_next = 2 * A_0 * A^{r} + 1




    # Pre-compute convolutions 
    S_conv = 3 * convolve(u_appro, u_appro, mode='full')
    
    Ar_expand = int(2 * Ar)
    Lr_expand = int(2 * Ar_expand + 1)


    # Expand each convolution matrix to the interval [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    offset_2 = int(A_next - Ar_expand)
    
    S_next = np.zeros((L_next, L_next))
    S_next[offset_2 : offset_2 + Lr_expand, offset_2 : offset_2 + Lr_expand] = S_conv



    # Prepare the 1D vectors for frequency coordinates
    ks = np.arange(-A_next, A_next + 1)
    N_grid, K_grid = np.meshgrid(ks, ks, indexing='ij')
    
    N_v = N_grid.flatten()
    K_v = K_grid.flatten()
    center = int(A_next)


    # Locate the indices of resonance points for four-point excision
    e_vec_pp = np.array([ torus_idx,  1])
    e_vec_mp = np.array([-torus_idx,  1])
    e_vec_pm = np.array([ torus_idx, -1])
    e_vec_mm = np.array([-torus_idx, -1])
    coords = np.column_stack((N_v, K_v))
    idx_e1 = np.where((coords == e_vec_pp).all(axis=1))[0][0]
    idx_e2 = np.where((coords == e_vec_mp).all(axis=1))[0][0]
    idx_e3 = np.where((coords == e_vec_pm).all(axis=1))[0][0]
    idx_e4 = np.where((coords == e_vec_mm).all(axis=1))[0][0]
    idx_drop = [idx_e1, idx_e2, idx_e3, idx_e4]


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
            T_full[mask_d, col] = eps * S_next[dn[mask_d] + center, dk[mask_d] + center]
            
    # At this point, T_full is exactly the matrix (eps * S)




    # --- Step 2: Add frequency derivative part B in-place ---
    offset_1 = int(A_next - Ar)
    u_appro_expand = np.zeros((L_next, L_next))  
    u_appro_expand[offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr] = u_appro
    
    V_row = -2 * mu_appro * (K_v)**2 * u_appro_expand.flatten()
    V_col_eps = T_full[idx_e1, :].copy()


    # Add B to T_full column by column
    for col in range(L_next**2):
        T_full[:, col] += (1 / (2 * a * mu_appro)) * V_row * V_col_eps[col]




    # --- Step 3: Add diagonal part D directly to the diagonal of T_full ---
    diag_vec = -(mu_appro * K_v)**2 + mu(N_v)**2
    np.fill_diagonal(T_full, T_full.diagonal() + diag_vec)




    # --- Step 4: Perform four-point excision on the finalized matrix ---
    T_full = np.delete(T_full, idx_drop, axis=0) 
    T_full = np.delete(T_full, idx_drop, axis=1) 
    
    T = T_full

    return T