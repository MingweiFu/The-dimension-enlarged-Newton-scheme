%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This function solves the Q-equations.                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro: vector of z coefficients                                  %%%
%%% eps:     perturbation quantity                                     %%%
%%% mu:      initial frequency                                         %%%
%%% a:       initial coefficient                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% mu_appro:        the approximate frequency                         %%%
%%% H1_partial_barz: convolution vector                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [mu_appro, H1_partial_barz] = Q_eqn(z_appro, eps, mu, a)


% settings of parameters
z_appro_bar = rot90(z_appro, 2);  % \bar{z}(k) = z(-k)
q = z_appro + z_appro_bar;        % q(k) = z(k) + \bar{z}(k) 

L = size(z_appro, 1);
A = (L - 1) / 2;
A_new = 3 * A;  % The first derivative of H1 is cubic, requiring triple convolution; thus, the support expands to 3A

H1_partial_barz = conv(q, conv(q, q)) / 4;                 
mu_appro = mu + eps * H1_partial_barz(1 + A_new + 1) / a;  

end


