%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/06  Mingwei Fu                                             %%%
%%% This function solves the Q-equations.                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro: tensor of z_1 and z_2 coefficients                        %%%
%%% eps:     perturbation quantity                                     %%%
%%% mu:      initial frequencies                                       %%%
%%% a:       initial coefficients                                      %%%
%%% n:       dimension of torus                                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% mu_appro:        the approximate frequencies                       %%%
%%% H1_partial_bar1: convolution matrix                                %%%
%%% H2_partial_bar2: convolution matrix                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [mu_appro, H1_partial_bar1, H1_partial_bar2] = Q_eqn(z_appro, eps, mu, a, n)


% settings of parameters
L = size(z_appro(:, :, 1), 1);
A = (L - 1) / 2;
A_new = 2 * A;

z_appro_bar = zeros(L, L, n);
for j = 1 : n
    z_appro_bar(:, :, j) = rot90(z_appro(:, :, j), 2);
end

q_1 = z_appro(:, :, 1) + z_appro_bar(:, :, 1);  % q_1(k) = z_1(k) + \bar{z}_1(k)
q_2 = z_appro(:, :, 2) + z_appro_bar(:, :, 2);  % q_2(k) = z_2(k) + \bar{z}_2(k)

mu_appro = zeros(n, 1);

H1_partial_bar1 = ( 1 / sqrt(2) ) * conv2(q_1, q_2);    
mu_appro(1,1) = mu(1,1) + eps * H1_partial_bar1(1 + A_new + 1, 0 + A_new + 1) / a(1,1);  % Update the first frequency component 

H1_partial_bar2 = ( 1 / (2 * sqrt(2)) ) * ( conv2(q_1, q_1) - conv2(q_2, q_2) );    
mu_appro(2,1) = mu(2,1) + eps * H1_partial_bar2(0 + A_new + 1, 1 + A_new + 1) / a(2,1);  % Update the second frequency component

end


