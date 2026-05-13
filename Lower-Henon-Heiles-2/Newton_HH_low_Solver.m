%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/09  Mingwei Fu                                             %%%
%%% This function is the Newton algorithm that solves the Henon-Heiles %%%
%%% system of lower dimensional quasi-periodic case for both of the    %%%
%%% Fourier coefficients and the frequencies.                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% A:         truncation parameter                                    %%%
%%% eps:       perturbation quantity                                   %%%
%%% mu:        initial frequencies (dim = n)                           %%%
%%% a:         initial coefficients (dim = m = 1)                      %%%
%%% R_max:     max iteration                                           %%%
%%% tol:       tolerance                                               %%%
%%% n:         dimension of torus                                      %%%
%%% torus_idx: index of the chosen torus                               %%%
%%% z_ini:     initial approximate solution                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% z_history:   approximate coefficients                              %%%
%%% fre_history: approximate frequencies                               %%%
%%% res_history: residual ||F^{(r)}|| and difference                   %%%
%%%              ||z^{(r+1)} - z^{(r)}||                               %%%
%%% r:           real iteration steps                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [z_history, fre_history, res_history, r] = Newton_HH_low_Solver(A, eps, mu, a, R_max, tol, n, torus_idx, z_ini)


m = size(torus_idx, 1);   % m = 1 in fact

z_history   = cell(R_max+1, 1);
fre_history = zeros(m, R_max+1);
res_history = zeros(2, R_max+1);

z_history{1, 1} = z_ini;




% Start Newton iteration loop 
r = 1;
while r <= R_max 
    % --- Step 1: Frequency Correction (Q-equation)  ---
    % Calculate the approximate frequency mu_r
    [mu_next, ~]   = Q_eqn(z_history{r, 1}, eps, mu, a, n, torus_idx);
    fre_history(r) = mu_next;
    
    % --- Step 2: calculates the residual of P-eqns (P-equation)  ---
    % Calculate the residual Fr
    [~, ~, ~, Fr_vec_red] = P_eqn_calcu(z_history{r, 1}, fre_history(r), mu, eps, n, torus_idx, A);
    
    curr_eqn_res = norm(Fr_vec_red, 2);
    res_history(1, r) = curr_eqn_res;  

    % --- Step 3: Newton Update  ---
    % Construct linearized operator T_{A^{r+1}} with Q-eqn indices removed
    [T_r, ~, ~, ~] = T_construct(z_history{r,1}, fre_history(r), mu, eps, a, n, torus_idx, A);

    % Solve the Newton equation: T_r * dy = -Fr
    dz_vec_red = - T_r \ Fr_vec_red;

    curr_sol_diff = norm(dz_vec_red,2);
    res_history(2, r) = curr_sol_diff;  

    % Restore the Newton update to matrix form via zero-padding
    dz_matrix = Vectorization_Process_Inverse(dz_vec_red, A^(r+1), n, torus_idx);

    % Form the next approximate solution
    z_next = zeros(2*A^(r+1)+1, n);
    for j = 1 : n
        z_r_expand = Vector_Expand_Padding(z_history{r,1}(:, j), A^r, A^(r+1));
        z_next(:, j) = z_r_expand + dz_matrix(:, j);
    end

    z_history{r+1, 1} = z_next;
    r = r+1;


    % --- Step 4: Termination checking  ---
    if curr_eqn_res < tol || curr_sol_diff < tol, break; end
end

% Calculate frequency for the final step
[mu_next, ~]   = Q_eqn(z_history{r, 1}, eps, mu, a, n, torus_idx);
fre_history(:, r) = mu_next;

% Calculate residual Fr for the final step
[~, ~, ~, Fr_vec_red] = P_eqn_calcu(z_history{r, 1}, fre_history(r), mu, eps, n, torus_idx, A);

curr_eqn_res = norm(Fr_vec_red, 2);
res_history(1, r) = curr_eqn_res;  

r = r-1;


end