%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/21  Mingwei Fu                                             %%%
%%% This function calculates the P-equations and return a vector.      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:   tensor of z_j (j = 1:n) coefficients                    %%%
%%% mu_appro:  approximate frequencies (dim = m = 2)                   %%%
%%% mu:        initial frequencies (dim = n)                           %%%
%%% eps:       perturbation quantity                                   %%%
%%% n:         dimension of torus                                      %%%
%%% torus_idx: indices of these chosen tori                            %%%
%%% A:         truncation parameter                                    %%%
%%% V:         the coefficient tensor of nonlinear terms               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% Fr_tensor_ori:    P-equations values in forms of tensor of size    %%%
%%%                   (Ar*A, Ar*A, n) without deleting the Q-eqns      %%%
%%% Fr_tensor_nan:    P-equations values in forms of tensor of size    %%%
%%%                   (Ar*A, Ar*A, n) setting nan at the Q-equations   %%%
%%% Fr_comp_full_nan: n vectors rearranged before deleting Q-eqns      %%%
%%%                   terms, setting nan at the Q-eqn terrms           %%%
%%% Fr_vec_tot_nan:   the whole vectors before deleting Q-eqn terms,   %%%
%%%                   setting nan at the Q-eqn terms                   %%%
%%% Fr_vec_red:       the whole vectors after deleting Q-equations     %%%
%%%                   in forms of a rearranged vector of length        %%%
%%%                   ( (2*A^{r+1}+1)^2 ) * n - m                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [Fr_tensor_ori, Fr_tensor_nan, Fr_comp_full_nan, Fr_vec_tot_nan, Fr_vec_red] = P_eqn_calcu(z_appro, mu_appro, mu, eps, n, torus_idx, A, V)


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
q = cell(n, 1);  
for j = 1 : n
    q{j, 1} = z_appro(:, :, j) + z_appro_bar(:, :, j);
end

Ar_expand = 3 * Ar;             % Ar_expand = 3 * A^r
Lr_expand = 2 * Ar_expand + 1;  % Lr_expand = 2 * (3A^{r}) + 1

H1_partial_barz = zeros(Lr_expand, Lr_expand, n); 
for j = 1 : n
    H1_partial_barj = zeros(Lr_expand, Lr_expand);

    for j2 = 1 : n
        for j3 = 1 : n
            conv_2_3 = conv2(q{j2}, q{j3});  
            for j4 = 1 : n
                if V(j, j2, j3, j4) == 0, continue; end

                conv_2_3_4 = conv2(conv_2_3, q{j4});

                H1_partial_barj = H1_partial_barj + 4 * V(j, j2, j3, j4) * conv_2_3_4;
            end
        end
    end

    H1_partial_barz(:, :, j) = H1_partial_barj;
end


% Expand the nonlinear terms to [-A^{r+1}, A^{r+1}]^2 and pad with zeros
offset = A_next - Ar_expand;
H1_partial_barz_next = zeros(L_next, L_next, n);

for j = 1 : n
    H1_partial_barz_next(offset + 1 : offset + Lr_expand, offset + 1 : offset + Lr_expand, j) = H1_partial_barz(:, :, j);  % k \in [-A^{r+1}, A^{r+1}]^2
end




% Calculate the linear part of the P-equation
ks_curr    = -Ar : Ar;  
[K1, K2]   = ndgrid(ks_curr, ks_curr);  
inner_prod = K1 * mu_appro(1) + K2 * mu_appro(2);  

Linear_z = zeros(Lr, Lr, n);  
for j = 1 : n
    Linear_z(:, :, j) = (-inner_prod + mu(j)) .* z_appro(:, :, j);  
end


% Expand the linear part to [-A^{r+1}, A^{r+1}]^2 and pad with zeros
offset_1 = A_next - Ar;
Linear_z_next     = zeros(L_next, L_next, n);

for j = 1 : n
    Linear_z_next(offset_1 + 1 : offset_1 + Lr, offset_1 + 1 : offset_1 + Lr, j)     = Linear_z(:, :, j);      % k \in [-A^{r+1}, A^{r+1}]^2
end




% Combine the linear and nonlinear parts on [-A^{r+1}, A^{r+1}]^2
Fr_tensor_ori = zeros(L_next, L_next, n);
for j = 1 : n
    Fr_tensor_ori(:, :, j) = Linear_z_next(:, :, j) + eps * H1_partial_barz_next(:, :, j);
end


% Mark the resonance set positions in the P-equation and reduce
[Fr_tensor_nan, Fr_comp_full_nan, Fr_vec_tot_nan, Fr_vec_red] = Vectorization_Process(Fr_tensor_ori, A_next, n, torus_idx);


end