import numpy as np
from scipy.signal import convolve

def Q_eqn(u_appro, A, eps, mu, a, torus_idx):
    """
    2026/06/29  Mingwei Fu
    This function solves the Q-equations for 1d-NLW with periodic b.c.
    for one chosen torus.

    Input:
    u_appro:   matrix of u(n,k) with (n,k) \in
               [-A_0*A^{r-1}, A_0*A^{r-1}]^{1+b}
    A:         expanding parameter
    eps:       perturbation quantity
    mu:        initial frequency function @(n) mu(n) = (|n|^2 + rho)^(1/2)
    a:         initial coefficients (dim = b = 1)
    torus_idx: index of the chosen torus

    Output:
    mu_appro:     the approximate frequency
    H1_partial_u: convolution matrix
    """

    # settings of parameters
    L_curr = u_appro.shape[0]
    A_curr = int((L_curr - 1) / 2)
    A_new  = int(A * A_curr)
    
    S_term_p = np.array([torus_idx, 1])


    # calculate the Q-eqns
    uu_conv = convolve(u_appro, u_appro, mode='full')
    H1_partial_u = convolve(uu_conv, u_appro, mode='full')

    row_idx = int(S_term_p[0] + A_new)
    col_idx = int(S_term_p[1] + A_new)

    mu_appro = np.sqrt(mu(torus_idx)**2 + (eps / a) * H1_partial_u[row_idx, col_idx])

    return mu_appro, H1_partial_u