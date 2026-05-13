%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/20  Mingwei Fu                                             %%%
%%% This function solves the Q-equations for FPU model with n mass     %%%
%%% points and two chosen basic tori (m=2).                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:   tensor of z_j (j = 1:n) coefficients                    %%%
%%% eps:       perturbation quantity                                   %%%
%%% mu:        initial frequencies (dim = n)                           %%%
%%% a:         initial coefficients (dim = m = 2)                      %%%
%%% n:         dimension of FPU model                                  %%%
%%% torus_idx: indices of these chosen tori                            %%%
%%% V:         the coefficient tensor of first-order derivatives       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% mu_appro:        the approximate frequencies                       %%%
%%% H1_partial_barz: convolution matrix                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [mu_appro, H1_partial_barz] = Q_eqn(z_appro, eps, mu, a, n, torus_idx, V)


% settings of parameters
m = size(torus_idx, 1);
L = size(z_appro(:, :, 1), 1);
A = (L - 1) / 2;
A_new = 3 * A; 
L_new = 2 * A_new + 1;

% calculate the Q-eqns
z_appro_bar = zeros(L, L, n);
for j = 1 : n
    z_appro_bar(:, :, j) = rot90(z_appro(:, :, j), 2);
end

q = cell(n, 1); 
for j = 1 : n
    q{j, 1} = z_appro(:, :, j) + z_appro_bar(:, :, j);
end

mu_appro = zeros(m, 1);

H1_partial_barz = zeros(L_new, L_new, m);
for j = 1 : m
    H1_partial_barj = zeros(L_new, L_new);

    for j2 = 1 : n
        for j3 = 1 : n
            conv_2_3 = conv2(q{j2}, q{j3}); 
            for j4 = 1 : n
                if V(torus_idx(j), j2, j3, j4) == 0, continue; end

                conv_2_3_4 = conv2(conv_2_3, q{j4});

                H1_partial_barj = H1_partial_barj + 4 * V(torus_idx(j), j2, j3, j4) * conv_2_3_4;
            end
        end
    end

    H1_partial_barz(:, :, j) = H1_partial_barj;
    ej = zeros(1, m);
    ej(j) = 1;
    mu_appro(j, 1) = mu(torus_idx(j)) + (eps / a(j)) * H1_partial_barj(ej(1) + A_new + 1, ej(2) + A_new + 1);
end


end


