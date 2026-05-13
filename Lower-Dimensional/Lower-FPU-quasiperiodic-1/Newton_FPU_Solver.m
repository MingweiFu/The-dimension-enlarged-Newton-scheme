%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/20  Mingwei Fu                                             %%%
%%% This function is the Newton algorithm that solves the FPU model    %%%
%%% with number of mass point n(=3), for two chosen basic tori.        %%%
%%% Fourier coefficients and frequencies are both obtained.            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% A:         truncation parameter                                    %%%
%%% eps:       perturbation quantity                                   %%%
%%% mu:        initial frequencies (dim = n)                           %%%
%%% a:         initial coefficients (dim = m = 2)                      %%%
%%% R_max:     max iteration                                           %%%
%%% tol:       tolerance                                               %%%
%%% n:         dimension of FPU model                                  %%%
%%% torus_idx: indices of these chosen tori                            %%%
%%% z_ini:     initial approximate solution                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% z_history:   approximate coefficients                              %%%
%%% fre_history: approximate frequencies                               %%%
%%% res_history: residual ||F^{(r)}|| and difference                   %%%
%%%              ||z^{(r+1)} - z^{(r)}||                               %%%
%%% r:           real iteration steps                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [z_history, fre_history, res_history, r] = Newton_FPU_Solver(A, eps, mu, a, R_max, tol, n, torus_idx, z_ini)


m = size(torus_idx, 1);  % m = 2 in fact

z_history   = cell(R_max+1, 1);
fre_history = zeros(m, R_max+1);
res_history = zeros(2, R_max+1);

z_history{1, 1} = z_ini;


% Calculate the coefficient tensor V(j1, j2, j3, j4) for the nonlinear terms
V = zeros(n, n, n, n);
n_total = n + 1;  

for j1 = 1 : n
    for j2 = 1 : n
        for j3 = 1 : n
            for j4 = 1 : n
                
                % Calculate the term T(j1, j2, j3, j4)
                T = 0;
                for l = 0 : n
                    term = cos(pi * j1 * (l + 0.5) / n_total) * ...
                           cos(pi * j2 * (l + 0.5) / n_total) * ...
                           cos(pi * j3 * (l + 0.5) / n_total) * ...
                           cos(pi * j4 * (l + 0.5) / n_total);
                    T = T + term;
                end
                
                % Calculate the V coefficients
                coef = sqrt(mu(j1) * mu(j2) * mu(j3) * mu(j4)) / (4 * n_total^2);
                V(j1, j2, j3, j4) = coef * T;
                
            end
        end
    end
end




% Start Newton iteration loop
r = 1;
while r <= R_max 
    % --- Step 1: Frequency Correction (Q-equation)  ---
    % Calculate the approximate frequency mu_r
    [mu_next, ~] = Q_eqn(z_history{r, 1}, eps, mu, a, n, torus_idx, V);
    fre_history(:, r) = mu_next;
       
    
    % --- Step 2: calculates the residual of P-eqns (P-equation)  ---
    % Calculate the residual Fr 
    [~, ~, ~, ~, Fr_vec_red] = P_eqn_calcu(z_history{r, 1}, fre_history(:, r), mu, eps, n, torus_idx, A, V);
    
    curr_eqn_res = norm(Fr_vec_red, 2);
    res_history(1, r) = curr_eqn_res;  


    % --- Step 3: Newton Update  ---
    % Construct linearized operator T_{A^{r+1}} with Q-eqn indices removed
    [T_r, ~, ~, ~] = T_construct(z_history{r,1}, fre_history(:, r), mu, eps, a, n, torus_idx, A, V);

    % Solve the Newton equation: T_r * dy = -Fr
    dz_vec_red = - T_r \ Fr_vec_red;

    curr_sol_diff = norm(dz_vec_red,2);
    res_history(2, r) = curr_sol_diff; 

    % Restore the Newton update to matrix form via zero-padding
    dz_tensor = Vectorization_Process_Inverse(dz_vec_red, A^(r+1), n, torus_idx);

    % Form the next approximate solution
    z_next = zeros(2*A^(r+1)+1, 2*A^(r+1)+1, n);
    for j = 1 : n
        z_r_expand = Matrix_Expand_Padding(z_history{r,1}(:, :, j), A^r, A^(r+1));
        z_next(:, :, j) = z_r_expand + dz_tensor(:, :, j);
    end

    z_history{r+1, 1} = z_next;
    r = r+1;


    % --- Step 4: Termination checking  ---
    if curr_eqn_res < tol || curr_sol_diff < tol, break; end
end

% Calculate frequency for the final step
[mu_next, ~] = Q_eqn(z_history{r, 1}, eps, mu, a, n, torus_idx, V);
fre_history(:, r) = mu_next;

% Calculate residual Fr for the final step
[~, ~, ~, ~, Fr_vec_red] = P_eqn_calcu(z_history{r, 1}, fre_history(:, r), mu, eps, n, torus_idx, A, V);

curr_eqn_res = norm(Fr_vec_red, 2);
res_history(1, r) = curr_eqn_res; 

r = r-1;


end