%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This function calculates the P-equations and return a vector.      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:  vector of z coefficients                                 %%%
%%% mu_appro: approximate frequency                                    %%%
%%% mu:       initial frequency                                        %%%
%%% eps:      perturbation quantity                                    %%%
%%% A:        truncation parameter                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% Fr_vec_ori: P-equations values in forms of matrix of size          %%%
%%%             (Ar*A, n) withour deleting the Q-equations (n=2)       %%%
%%% Fr_vec_nan: the whole vectors before deleting Q-equation terms     %%%
%%% Fr_vec_red: the whole vectors after deleting Q-equation terms      %%%
%%%             in forms of a rearranged vector of length              %%%
%%%             ( (2*A^{r+1}+1)^2 ) * n - n                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [Fr_vec_ori, Fr_vec_nan, Fr_vec_red] = P_eqn_calcu(z_appro, mu_appro, mu, eps, A)


% settings
z_appro_bar = rot90(z_appro, 2);  % \bar{z}(k) = z(-k)

Lr     = size(z_appro, 1);  % Lr = 2 * A^r + 1
Ar     = (Lr - 1) / 2;      % Ar = A^r
A_next = Ar * A;            % A_next = A^{r+1}
L_next = 2 * A_next + 1;    % L_next = 2 * A^{r+1} + 1




% Calculate the convolution for the nonlinear terms of the P-equation
q = z_appro + z_appro_bar;  % q(k) = z(k) + \bar{z}(k)
H1_partial_barz = conv(q, conv(q, q)) / 4;

Lr_expand = size(H1_partial_barz, 1);  % Lr_expand = 2 * (3 * A^r) + 1
Ar_expand = (Lr_expand - 1) / 2;       % Ar_expand = 3 * A^r


% Expand the nonlinear terms to [-A^{r+1}, A^{r+1}] and pad with zeros
offset = A_next - Ar_expand;
H1_partial_barz_next = zeros(L_next, 1);                                  
H1_partial_barz_next(offset + 1 : offset + Lr_expand) = H1_partial_barz;  




% Calculate the linear part of the P-equation
ks_curr = (-Ar : Ar)';  
inner_prod = ks_curr * mu_appro;  
Linear_z     = (-inner_prod + mu) .* z_appro;


% Expand the linear part to [-A^{r+1}, A^{r+1}] and pad with zeros
offset_1 = A_next - Ar;
Linear_z_next = zeros(L_next, 1);                       
Linear_z_next(offset_1 + 1 : offset_1 + Lr) = Linear_z;  




% Combine the linear and nonlinear parts on [-A^{r+1}, A^{r+1}]
Fr_vec_ori = Linear_z_next + eps * H1_partial_barz_next;

% Mark the resonance set positions in the P-equation and reduce
[Fr_vec_nan, Fr_vec_red] = Vectorization_Process(Fr_vec_ori, A_next);


end