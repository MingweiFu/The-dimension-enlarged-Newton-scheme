import numpy as np
from scipy.linalg import solve

from Q_eqn import Q_eqn
from P_eqn_calcu import P_eqn_calcu
from T_construct_opt import T_construct_opt
from Tensor_Expand_Padding import Tensor_Expand_Padding
from Vectorization_Process_Inverse import Vectorization_Process_Inverse

def Newton_2d_NLS_Solver(A_0, A, eps, mu, a, R_max, tol, torus_idx, u_ini):
    """
    2026/05/26  Mingwei Fu
    This function is the Newton algorithm that solves the 1d-NLS
    with dirichlet b.c., for two chosen tori.
    Fourier coefficients and frequencies are both obtained.

    Input:
    A_0:       initial support parameter
    A:         expanding parameter
    eps:       perturbation quantity
    mu:        initial frequency function @(n) mu(n) = |n|^2 + m
    a:         initial coefficients (dim = b = 2)
    R_max:     max iteration
    tol:       tolerance
    torus_idx: indices of the chosen tori
    u_ini:     initial approximate solution

    Output:
    u_history:   approximate coefficients 
    fre_history: approximate frequencies 
    res_history: residual ||F^{(r)}|| and difference ||u^{(r+1)} - u^{(r)}||
    r:           real iteration steps
    """

    b = np.atleast_1d(torus_idx).shape[0]  # b = 2 in fact

    u_history = [u_ini]
    fre_history = np.zeros((b, R_max + 1))
    res_history = np.zeros((2, R_max + 1))

    r = 0

    while r < R_max:
        # --- Step 1: Frequency Correction (Q-equation) ---
        # Calculate the approximate frequency mu_r
        mu_next, _ = Q_eqn(u_history[r], A, eps, mu, a, torus_idx)
        fre_history[:, r] = mu_next


        # --- Step 2: Calculates the residual of P-eqns (P-equation) ---
        # Calculate the residual Fr
        _, _, _, Fr_vec_red = P_eqn_calcu(u_history[r], fre_history[:, r], mu, eps, torus_idx, A)

        curr_eqn_res = np.linalg.norm(Fr_vec_red, ord=2)
        res_history[0, r] = curr_eqn_res


        # --- Step 3: Newton Update ---
        # Construct linearized operator T_{A_0*A^{r}} with Q-eqn indices removed
        # T_r, _, _, _ = T_construct(u_history[r], fre_history[:, r], mu, eps, a, torus_idx, A)
        
        T_r = T_construct_opt(u_history[r], fre_history[:, r], mu, eps, a, torus_idx, A)

        # Solve the Newton equation: T_r * dy = -Fr
        # du_vec_red = -np.linalg.solve(T_r, Fr_vec_red)
        
        du_vec_red = -solve(T_r, Fr_vec_red, overwrite_a=True, overwrite_b=True, check_finite=False)

        curr_sol_diff = np.linalg.norm(du_vec_red, ord=2)
        res_history[1, r] = curr_sol_diff

        # Restore the Newton update to tensor form via zero-padding
        du_tensor = Vectorization_Process_Inverse(du_vec_red, int(A_0 * A ** (r+1)), torus_idx)

        # Form the next approximate solution
        u_r_expand = Tensor_Expand_Padding(u_history[r], int(A_0 * A**r), int(A_0 * A ** (r+1)))
        u_next = u_r_expand + du_tensor

        u_history.append(u_next)
        r += 1




        # --- Step 4: Termination checking ---
        if curr_eqn_res < tol or curr_sol_diff < tol:
            break

    # Calculate frequency for the final step
    mu_next, _ = Q_eqn(u_history[r], A, eps, mu, a, torus_idx)
    fre_history[:, r] = mu_next

    # Calculate residual Fr for the final step
    _, _, _, Fr_vec_red = P_eqn_calcu(u_history[r], fre_history[:, r], mu, eps, torus_idx, A)

    curr_eqn_res = np.linalg.norm(Fr_vec_red, ord=2)
    res_history[0, r] = curr_eqn_res

    fre_history = fre_history[:, :r+1]
    res_history = res_history[:, :r+1]

    return u_history, fre_history, res_history, r