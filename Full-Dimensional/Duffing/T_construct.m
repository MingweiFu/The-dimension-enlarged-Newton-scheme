%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This function constructs the linearized operator                   %%%
%%% T_{A^{r+1}}, which is size ( 2A^{r+1} + 1 - 1 ) \times             %%%
%%% ( 2A^{r+1} + 1 - 1 ).                                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:  vector of z coefficients                                 %%%
%%% mu_appro: approximate frequency                                    %%%
%%% mu:       initial frequency                                        %%%
%%% eps:      perturbation quantity                                    %%%
%%% A:        truncation parameter                                     %%%
%%% a:        initial coefficient                                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% T: linearized matrix of size ( 2A^{r+1} + 1 - 1 ) * 2 ×            %%%
%%%    ( 2A^{r+1} + 1 - 1 ) * 2, with resonant entries reduced         %%%
%%% D: diagonal part of the linearized matrix                          %%%
%%% S: the linearized matrix of the nonlinear term                     %%%
%%% B: the additional matrix arises from the difference of approximate %%%
%%%    frequencies                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [T, D, S, B] = T_construct(z_appro, mu_appro, mu, eps, A, a)


% settings
z_appro_bar = rot90(z_appro, 2);  % \bar{z}(k) = z(-k)

Lr     = size(z_appro, 1);  % Lr = 2 * A^r + 1
Ar     = (Lr - 1) / 2;      % Ar = A^r
A_next = Ar * A;            % A_next = A^{r+1}
L_next = 2 * A_next + 1;    % L_next = 2 * A^{r+1} + 1




% --- Step 1: Construct the diagonal part D of operator T ---
ks_curr       = (-A_next : A_next)';  
inner_prod    = ks_curr * mu_appro; 
D_part_vector = -inner_prod + mu;  

[~, D_vec_red] = Vectorization_Process(D_part_vector, A_next);  
D = diag(D_vec_red);                                          


% Expand z_appro and z_appro_bar to [-A^{r+1}, A^{r+1}] and pad with zeros
offset_1 = A_next - Ar;

z_appro_expand = zeros(L_next, 1);  
z_appro_expand(offset_1 + 1 : offset_1 + Lr, 1) = z_appro(:, 1);




% --- Step 2: Construct the nonlinear perturbation part S of T ---
q = z_appro(:, 1) + z_appro_bar(:, 1);  % q(k) = z(k) + \bar{z}(k)
phi_hat = (3/4) * conv(q, q);
center = 2 * Ar + 1;                    % Index of frequency 0 in phi_hat

% Construct the S1 matrix (frequency shift k - k') and fill using a mask
ks = (-A_next : A_next)'; 
[KK_prime, KK] = meshgrid(ks, ks);  

dk   = KK - KK_prime;                 
mask = (abs(dk) <= 2 * Ar);         

S1_full = zeros(L_next, L_next);
S1_full(mask) = phi_hat(dk(mask) + center);  

% Construct the S2 matrix (frequency sum k + k') and fill using a mask
dk_sum   = KK + KK_prime;             
mask_sum = (abs(dk_sum) <= 2 * Ar);  

S2_full = zeros(L_next, L_next);
S2_full(mask_sum) = phi_hat(dk_sum(mask_sum) + center);

% Combine into the full-scale S matrix, S = S1 + S2 
S_full = S1_full + S2_full;

idx_1  = 1 + A_next + 1;  
S = S_full;
S(idx_1, :) = [];  % Remove the idx_1 row
S(:, idx_1) = [];  % Remove the idx_1 column




% --- Step 3: Construct the frequency derivative part B of T ---
V_row = -ks .* z_appro_expand;  
V_col = S_full(idx_1, :).';     

% Construct the B1 matrix (before excluding resonant indices)
B1_full = V_row * V_col.';

% Remove the resonant term k = 1 from B1
B1 = B1_full;
B1(idx_1, :) = [];   % Remove the idx_1 row
B1(:, idx_1) = [];   % Remove the idx_1 column
B = (eps / a) * B1;  



% --- Step 4: Assemble the complete linearized operator T ---
T = D + eps * S + B;

end