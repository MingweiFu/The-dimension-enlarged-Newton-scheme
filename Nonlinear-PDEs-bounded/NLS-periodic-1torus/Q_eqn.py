import numpy as np
from scipy.signal import convolve

def Q_eqn(u_appro, A, eps, mu, a, torus_idx):
    """
    2026/05/15  Mingwei Fu
    This function solves the Q-equations for 1d-NLS with dirichlet b.c.
    for one chosen torus.

    Input:
    u_appro:   matrix of u(n,k) with (n,k) \in
               [-A_0*A^{r-1}, A_0*A^{r-1}]^{1+b}
    A:         expanding parameter
    eps:       perturbation quantity
    mu:        initial frequency function @(n) mu(n) = |n|^2 + m
    a:         initial coefficients (dim = b = 1)
    torus_idx: index of the chosen torus

    Output:
    mu_appro:        the approximate frequency
    H1_partial_baru: convolution matrix
    """
    
    # settings of parameters
    L_curr = u_appro.shape[0]          
    A_curr = int((L_curr - 1) / 2)     
    A_new  = int(A * A_curr)
    S_term = np.array([torus_idx, 1]) 


    # calculate the Q-eqns
    u_appro_bar = np.flip(u_appro.copy())

    uu_conv = convolve(u_appro, u_appro, mode='full')
    H1_partial_baru = convolve(uu_conv, u_appro_bar, mode='full')

    row_idx = int(S_term[0] + A_new)
    col_idx = int(S_term[1] + A_new)

    mu_appro = mu(torus_idx) + (eps / a) * H1_partial_baru[row_idx, col_idx]

    return mu_appro, H1_partial_baru