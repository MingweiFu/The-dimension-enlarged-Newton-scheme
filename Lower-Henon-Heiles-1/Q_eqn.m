%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/09  Mingwei Fu                                             %%%
%%% This function solves the Q-equations.                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:   matrix of z_1 and z_2 coefficients                      %%%
%%% eps:       perturbation quantity                                   %%%
%%% mu:        initial frequency                                       %%%
%%% a:         initial coefficients                                    %%%
%%% n:         dimension of torus                                      %%%
%%% torus_idx: index of the chosen torus                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% mu_appro:        the approximate frequency                         %%%
%%% H1_partial_barj: convolution vector                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [mu_appro, H1_partial_barj] = Q_eqn(z_appro, eps, mu, a, n, torus_idx)


% settings of parameters
L      = size(z_appro(:, 1), 1);
A      = (L - 1) / 2;
A_conv = 2 * A;

% calculate the Q-eqns
z_appro_bar = zeros(L, n);
for j = 1 : n
    z_appro_bar(:, j) = rot90(z_appro(:, j), 2);
end

q_1 = z_appro(:, 1) + z_appro_bar(:, 1);  % q_1(k) = z_1(k) + \bar{z}_1(k)
q_2 = z_appro(:, 2) + z_appro_bar(:, 2);  % q_2(k) = z_2(k) + \bar{z}_2(k)

if torus_idx == 1
    H1_partial_barj = ( 1 / sqrt(2) ) * conv(q_1, q_2);
elseif torus_idx == 2
    H1_partial_barj = ( 1 / (2 * sqrt(2)) ) * ( conv(q_1, q_1) - conv(q_2, q_2) );
else
    error('Invalid torus_idx. Must be 1 or 2 for Henon-Heiles system.');
end

mu_appro = mu(torus_idx) + (eps / a) * H1_partial_barj(1 + A_conv + 1);


end


