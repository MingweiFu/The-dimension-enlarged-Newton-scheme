import numpy as np
from scipy.signal import convolve

from Vectorization_Process import Vectorization_Process

def P_eqn_calcu(u_appro, mu_appro, mu, eps, torus_idx, A):
    """
    2026/10/08  Mingwei Fu
    This function calculates the P-equations and returns a vector.

    Input:
    u_appro:   tensor of u(n,k1,k2) coefficients
    mu_appro:  approximate frequencies
    mu:        initial frequency function @(n) mu(n) = n^3 + m*n
    eps:       perturbation quantity
    torus_idx: indices of the chosen tori
    A:         expanding parameter

    Output:
    Fr_tensor_ori:  P-equations values in forms of tensor of size
                    (Ar*A, Ar*A, Ar*A) without deleting the Q-eqns and u(0,k1,k2) terms
    Fr_tensor_nan:  P-equations values in forms of tensor of size
                    (Ar*A, Ar*A, Ar*A) setting nan at the Q-equations and u(0,k1,k2) positions
    Fr_vec_tot_nan: the whole vectors with nan at Q-eqns and u(0,k1,k2) positions
    Fr_vec_red:     the whole vectors after deleting Q-equations and u(0,k1,k2) terms
                    in forms of a rearranged vector of length
                    ( (2*Ar*A+1)^(1+b) ) - 2b - (2*Ar*A+1)^b
    """

    # settings
    Lr     = u_appro.shape[0]     # Lr = 2 * A_0 * A^{r-1} + 1
    Ar     = int((Lr - 1) / 2)    # Ar = A_0 * A^{r-1}
    A_next = int(Ar * A)          # A_next = A_0 * A^{r}
    L_next = int(2 * A_next + 1)  # L_next = 2 * A_0 * A^{r} + 1




    # Calculate the convolution for the nonlinear terms of the P-equation
    Ar_expand = int(2 * Ar)
    Lr_expand = int(2 * Ar_expand + 1)

    H1_partial_u = convolve(u_appro, u_appro, mode="full")
    
    for i in range(Lr_expand):
        H1_partial_u[i, :, :] *= eps * (i - Ar_expand)


    # Expand the nonlinear terms to [-A_0*A^{r}, A_0*A^{r}]^{1+b} and pad with zeros
    offset = int(A_next - Ar_expand)
    nonlinear_next = np.zeros((L_next, L_next, L_next))

    nonlinear_next[offset : offset + Lr_expand, offset : offset + Lr_expand, offset : offset + Lr_expand] = H1_partial_u




    # Calculate the linear part of the P-equation
    ks_curr = np.arange(-Ar, Ar + 1)
    N, K1, K2 = np.meshgrid(ks_curr, ks_curr, ks_curr, indexing="ij")
    
    D_p = np.zeros((Lr, Lr, Lr))
    nonzero_n = (N != 0)
    D_p[nonzero_n] = mu_appro[0] * K1[nonzero_n] + mu_appro[1] * K2[nonzero_n] - mu(N[nonzero_n])

    Linear_u = D_p * u_appro


    # Expand the linear part to [-A_0*A^{r}, A_0*A^{r}]^{1+b} and pad with zeros
    offset_1 = int(A_next - Ar)
    Linear_u_next = np.zeros((L_next, L_next, L_next))

    Linear_u_next[offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr, offset_1 : offset_1 + Lr] = Linear_u




    # Combine the linear and nonlinear parts on [-A_0*A^{r}, A_0*A^{r}]^{1+b}
    Fr_tensor_ori = Linear_u_next + nonlinear_next
    
    n_zero_idx = int(A_next)
    Fr_tensor_ori[n_zero_idx, :, :] = 0


    # Mark the resonance and u(0,k1,k2) positions in the P-equation and reduce
    Fr_tensor_nan, Fr_vec_tot_nan, Fr_vec_red = Vectorization_Process(Fr_tensor_ori, A_next, torus_idx)

    return Fr_tensor_ori, Fr_tensor_nan, Fr_vec_tot_nan, Fr_vec_red