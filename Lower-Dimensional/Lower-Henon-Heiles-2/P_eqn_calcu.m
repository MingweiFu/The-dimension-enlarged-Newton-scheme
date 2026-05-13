%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/10  Mingwei Fu                                             %%%
%%% This function calculates the P-equations and return a vector.      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:   matrix of z_1 and z_2 coefficients                      %%%
%%% mu_appro:  approximate frequency                                   %%%
%%% mu:        initial frequencies                                     %%%
%%% eps:       perturbation quantity                                   %%%
%%% n:         dimension of torus                                      %%%
%%% torus_idx: index of the chosen torus                               %%%
%%% A:         truncation parameter                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% Fr_matrix_ori:  P-eqns values in forms of matrix of size           %%%
%%%                 (Ar*A, n) without deleting the Q-eqns              %%%
%%% Fr_matrix_nan:  P-eqns values in forms of matrix of size           %%%
%%%                 (Ar*A, n) setting nan at the Q-eqn terms           %%%
%%% Fr_vec_tot_nan: the whole vectors before deleting Q-eqn terms,     %%%
%%%                 setting nan at the Q-eqn terms                     %%%
%%% Fr_vec_red:     the whole vectors after deleting Q-eqn terms       %%%
%%%                 in forms of a rearranged vector of length          %%%
%%%                 ( (2*A^{r+1}+1)^2 ) * n - 1                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [Fr_matrix_ori, Fr_matrix_nan, Fr_vec_tot_nan, Fr_vec_red] = P_eqn_calcu(z_appro, mu_appro, mu, eps, n, torus_idx, A)


% settings
Lr     = size(z_appro(:, 1), 1);  % Lr = 2 * A^r + 1
Ar     = (Lr - 1) / 2;            % Ar = A^r
A_next = Ar * A;                  % A_next = A^{r+1}
L_next = 2 * A_next + 1;          % L_next = 2 * A^{r+1} + 1

% \bar{z}(k) = z(-k)
z_appro_bar = zeros(Lr, n);
for j = 1 : n
    z_appro_bar(:, j) = rot90(z_appro(:, j), 2);
end




% Calculate the convolution for the nonlinear terms of the P-equation
q_1 = z_appro(:, 1) + z_appro_bar(:, 1);  % q_1(k) = z_1(k) + \bar{z}_1(k)
q_2 = z_appro(:, 2) + z_appro_bar(:, 2);  % q_2(k) = z_2(k) + \bar{z}_2(k)

H1_partial_bar1 = ( 1 / sqrt(2) ) * conv(q_1, q_2);
H1_partial_bar2 = ( 1 / (2 * sqrt(2)) ) * ( conv(q_1, q_1) - conv(q_2, q_2) );

Lr_expand = size(H1_partial_bar1, 1);  % Lr_expand = 2 * (2 * A^r) + 1
Ar_expand = (Lr_expand - 1) / 2;       % Ar_expand = 2 * A^r


% Expand the nonlinear terms to [-A^{r+1}, A^{r+1}] and pad with zeros
offset_H = A_next - Ar_expand;

H1_partial_barz_next = zeros(L_next, n);
H1_partial_barz_next(offset_H + 1 : offset_H + Lr_expand, 1) = H1_partial_bar1;
H1_partial_barz_next(offset_H + 1 : offset_H + Lr_expand, 2) = H1_partial_bar2;




% Calculate the linear part of the P-equation
ks_curr    = (-Ar : Ar)';  
inner_prod = ks_curr * mu_appro; 

Linear_z = zeros(Lr, n);
for j = 1 : n
    Linear_z(:, j) = (-inner_prod + mu(j)) .* z_appro(:, j);
end


% Expand the linear part to [-A^{r+1}, A^{r+1}] and pad with zeros
offset_L = A_next - Ar;

Linear_z_next = zeros(L_next, n);
for j = 1 : n
    Linear_z_next(offset_L + 1 : offset_L + Lr, j) = Linear_z(:, j);
end




% Combine the linear and nonlinear parts on [-A^{r+1}, A^{r+1}]
Fr_matrix_ori = Linear_z_next + eps * H1_partial_barz_next;

% Mark the resonance set positions in the P-equation and reduce
[Fr_matrix_nan, Fr_vec_tot_nan, Fr_vec_red] = Vectorization_Process(Fr_matrix_ori, A_next, torus_idx);


end