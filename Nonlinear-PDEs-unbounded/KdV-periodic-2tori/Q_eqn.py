import numpy as np
from scipy.signal import convolve

def Q_eqn(u_appro, eps, mu, a, torus_idx):
    """
    2026/10/08  Mingwei Fu
    This function solves the Q-equations for 1d-KdV with periodic b.c.
    and zero mean value condition for two chosen tori.

    Input:
    u_appro:   tensor of u(n,k1,k2) with (n,k1,k2) \in
               [-A_0*A^{r-1}, A_0*A^{r-1}]^{1+b}
    eps:       perturbation quantity
    mu:        initial frequency function @(n) mu(n) = n^3 + m*n
    a:         initial coefficients (dim = b = 2)
    torus_idx: indices of the chosen tori

    Output:
    mu_appro:        the approximate frequencies
    H1_partial_baru: convolution tensor
    """

    # settings of parameters
    L_curr = u_appro.shape[0]
    A_curr = int((L_curr - 1) / 2)
    A_new  = int(2 * A_curr)
    S_term = np.array([[torus_idx[0], 1, 0], [torus_idx[1], 0, 1]])  # (n1, e1), (n2, e2)


    # calculate the Q-eqns
    H1_partial_u = convolve(u_appro, u_appro, mode="full")

    mu_appro = np.zeros(2)
    for j in range(2):
        idx_n = int(S_term[j, 0] + A_new)
        idx_k1 = int(S_term[j, 1] + A_new)
        idx_k2 = int(S_term[j, 2] + A_new)

        mu_appro[j] = mu(torus_idx[j]) - (eps * torus_idx[j] / a[j]) * H1_partial_u[idx_n, idx_k1, idx_k2]

    return mu_appro, H1_partial_u