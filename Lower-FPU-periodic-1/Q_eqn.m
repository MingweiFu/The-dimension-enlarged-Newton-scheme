%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/13  Mingwei Fu                                             %%%
%%% This function solves the Q-equations for FPU model with n mass     %%%
%%% points and one chosen basic torus (m=1).                           %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:     matrix of z_j (j = 1:n) coefficients                  %%%
%%% eps:         perturbation quantity                                 %%%
%%% mu:          initial frequencies (dim = n)                         %%%
%%% a:           initial coefficients (dim = m = 1)                    %%%
%%% n:           dimension of FPU model                                %%%
%%% torus_idx:   index of the chosen torus                             %%%
%%% V:           the coefficient tensor of first-order derivatives     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% mu_appro:        the approximate frequency                         %%%
%%% H1_partial_barj: convolution vector                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [mu_appro, H1_partial_barj] = Q_eqn(z_appro, eps, mu, a, n, torus_idx, V)


% settings of parameters
L = size(z_appro(:, 1), 1);
A = (L - 1) / 2;
A_conv = 3 * A; 
L_conv = 2 * A_conv + 1;

% calculate the Q-eqns
z_appro_bar = zeros(L, n);
for j = 1 : n
    z_appro_bar(:, j) = rot90(z_appro(:, j), 2);
end

q = cell(n, 1);  
for j = 1 : n
    q{j, 1} = z_appro(:, j) + z_appro_bar(:, j);
end

H1_partial_barj = zeros(L_conv, 1);
for j2 = 1 : n
    for j3 = 1 : n
        conv_2_3 = conv(q{j2}, q{j3});  
        for j4 = 1 : n
            if V(torus_idx, j2, j3, j4) == 0, continue; end

            conv_2_3_4 = conv(conv_2_3, q{j4});

            H1_partial_barj = H1_partial_barj + 4 * V(torus_idx, j2, j3, j4) * conv_2_3_4;
        end
    end
end

mu_appro = mu(torus_idx) + (eps / a) * H1_partial_barj(1 + A_conv + 1);



end


