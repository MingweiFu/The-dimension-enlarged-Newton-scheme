%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/07  Mingwei Fu                                             %%%
%%% This function calculates the P-equations and return a vector.      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:  tensor of z_1 and z_2 coefficients                       %%%
%%% mu_appro: approximate frequencies                                  %%%
%%% mu:       initial frequencies                                      %%%
%%% eps:      perturbation quantity                                    %%%
%%% n:        dimension of torus                                       %%%
%%% A:        truncation parameter                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% Fr_tensor_ori:    P-eqns values in forms of tensor of size         %%%
%%%                   (Ar*A, Ar*A, n) without deleting the Q-eqns      %%%
%%% Fr_tensor_nan:    P-eqns values in forms of tensor of size         %%%
%%%                   (Ar*A, Ar*A, n) setting nan at the Q-eqn terms   %%%
%%% Fr_comp_full_nan: n vectors rearranged before deleting Q-eqns      %%%
%%%                   terms, setting nan at the Q-eqn terrms           %%%
%%% Fr_vec_tot_nan:   the whole vectors before deleting Q-eqn terms,   %%%
%%%                   setting nan at the Q-eqn terms                   %%%
%%% Fr_vec_red:       the whole vectors after deleting Q-eqn terms     %%%
%%%                   in forms of a rearranged vector of length        %%%
%%%                   ( (2*A^{r+1}+1)^2 ) * n - n                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [Fr_tensor_ori, Fr_tensor_nan, Fr_comp_full_nan, Fr_vec_tot_nan, Fr_vec_red] = P_eqn_calcu(z_appro, mu_appro, mu, eps, n, A)


% settings
Lr     = size(z_appro(:, :, 1), 1);  % Lr = 2 * A^r + 1
Ar     = (Lr - 1) / 2;               % Ar = A^r
A_next = Ar * A;                     % A_next = A^{r+1}
L_next = 2 * A_next + 1;             % L_next = 2 * A^{r+1} + 1

% \bar{z}(k) = z(-k)
z_appro_bar = zeros(Lr, Lr, n);
for j = 1 : n
    z_appro_bar(:, :, j) = rot90(z_appro(:, :, j), 2);
end




% Calculate the convolution for the nonlinear terms of the P-equation
q_1 = z_appro(:, :, 1) + z_appro_bar(:, :, 1);  % q_1(k) = z_1(k) + \bar{z}_1(k)
q_2 = z_appro(:, :, 2) + z_appro_bar(:, :, 2);  % q_2(k) = z_2(k) + \bar{z}_2(k)

H1_partial_bar1 = ( 1 / sqrt(2) ) * conv2(q_1, q_2);
H1_partial_bar2 = ( 1 / (2 * sqrt(2)) ) * ( conv2(q_1, q_1) - conv2(q_2, q_2) );

Lr_expand = size(H1_partial_bar1, 1);  % Lr_expand = 2 * (2 * A^r) + 1
Ar_expand = (Lr_expand - 1) / 2;       % Ar_expand = 2 * A^r


% Expand the nonlinear terms to [-A^{r+1}, A^{r+1}]^2 and pad with zeros
offset = A_next - Ar_expand;
H1_partial_bar1_next = zeros(L_next, L_next); 
H1_partial_bar1_next(offset + 1 : offset + Lr_expand, offset + 1 : offset + Lr_expand) = H1_partial_bar1;  

H1_partial_bar2_next = zeros(L_next, L_next);  
H1_partial_bar2_next(offset + 1 : offset + Lr_expand, offset + 1 : offset + Lr_expand) = H1_partial_bar2;




% Calculate the linear part of the P-equation
ks_curr    = -Ar : Ar;  
[K1, K2]   = ndgrid(ks_curr, ks_curr);  
inner_prod = K1 * mu_appro(1) + K2 * mu_appro(2); 

Linear_z_1     = (-inner_prod + mu(1)) .* z_appro(:, :, 1);
Linear_z_2     = (-inner_prod + mu(2)) .* z_appro(:, :, 2);


% Expand the linear terms to [-A^{r+1}, A^{r+1}]^2 and pad with zeros
offset_1 = A_next - Ar;
Linear_z_1_next = zeros(L_next, L_next);  
Linear_z_1_next(offset_1 + 1 : offset_1 + Lr, offset_1 + 1 : offset_1 + Lr) = Linear_z_1;  

Linear_z_2_next = zeros(L_next, L_next);  
Linear_z_2_next(offset_1 + 1 : offset_1 + Lr, offset_1 + 1 : offset_1 + Lr) = Linear_z_2;




% Combine the linear and nonlinear parts on [-A^{r+1}, A^{r+1}]^2
Fr_tensor_ori = zeros(L_next, L_next, n);
Fr_tensor_ori(:, :, 1) = Linear_z_1_next + eps * H1_partial_bar1_next;
Fr_tensor_ori(:, :, 2) = Linear_z_2_next + eps * H1_partial_bar2_next;

% Mark the resonance set positions in the P-equation and reduce
[Fr_tensor_nan, Fr_comp_full_nan, Fr_vec_tot_nan, Fr_vec_red] = Vectorization_Process(Fr_tensor_ori, A_next, n);


end