%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/11  Mingwei Fu                                             %%%
%%% This function constructs the linearized operator                   %%%
%%% T_{A^{r+1}}, which is size ( (2A^{r+1}+1) * n - 1 ) \times         %%%
%%% ( (2A^{r+1}+1) * n - 1 ).                                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:   matrix of z_1 and z_2 coefficients                      %%%
%%% mu_appro:  approximate frequency                                   %%%
%%% mu:        initial frequencies                                     %%%
%%% eps:       perturbation quantity                                   %%%
%%% a:         initial coefficients                                    %%%
%%% n:         dimension of torus                                      %%%
%%% torus_idx: index of the chosen torus                               %%%
%%% A:         truncation parameter                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% T: linearized matrix of size ( (2A^{r+1}+1) * n - 2 ) ×            %%%
%%%    ( (2A^{r+1}+1) * n - 2 ), with resonant entries reduced         %%%
%%% D: diagonal part of the linearized matrix                          %%%
%%% S: the linearized matrix of the nonlinear term                     %%%
%%% B: the additional matrix arises from the difference of approximate %%%
%%%    frequencies                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [T, D, S, B] = T_construct(z_appro, mu_appro, mu, eps, a, n, torus_idx, A)


% settings
Lr     = size(z_appro(:, 1), 1);  % Lr = 2 * A^r + 1
Ar     = (Lr - 1) / 2;            % Ar = A^r
A_next = Ar * A;                  % A_next = A^{r+1}
L_next = 2 * A_next + 1;          % L_next = 2 * A^{r+1} + 1

z_appro_bar = zeros(Lr, n);
for j = 1 : n
    z_appro_bar(:, j) = rot90(z_appro(:, j), 2);
end




% --- Step 1: Construct the diagonal part D of operator T ---
ks_curr    = (-A_next : A_next)';  
inner_prod = ks_curr * mu_appro;

D_part_matrix = zeros(L_next, n); 
for j = 1 : n
    D_part_matrix(:, j) = -inner_prod + mu(j);
end

[~, ~, D_vec_red] = Vectorization_Process(D_part_matrix, A_next, torus_idx); 
D = diag(D_vec_red);  


% Expand z_appro and z_appro_bar to [-A^{r+1}, A^{r+1}] and pad with zeros
offset_1 = A_next - Ar;

z_appro_expand     = zeros(L_next, n);  
z_appro_bar_expand = zeros(L_next, n);  

for j = 1 : n
    z_appro_expand(offset_1 + 1 : offset_1 + Lr, j)     = z_appro(:, j);
    z_appro_bar_expand(offset_1 + 1 : offset_1 + Lr, j) = z_appro_bar(:, j);
end




% --- Step 2: Construct the nonlinear perturbation part S of T ---
q_expand = z_appro_expand + z_appro_bar_expand;  % q = z + bar_z

% The second derivative coefficients phi_hat
phi_hat       = cell(n, n);
phi_hat{1, 1} = ( 1 / sqrt(2) ) * q_expand(:, 2);              
phi_hat{1, 2} = ( 1 / sqrt(2) ) * q_expand(:, 1);         
phi_hat{2, 1} = ( 1 / sqrt(2) ) * q_expand(:, 1);
phi_hat{2, 2} = (-1 / sqrt(2) ) * q_expand(:, 2);       

% Prepare the linear index offset mapping for the 1D frequency grid
[K, K_prime] = ndgrid(ks_curr, ks_curr);

dk_diff = K - K_prime; 
dk_sum  = K + K_prime; 

% Identify masks for displacements within the valid support set [-Ar, Ar]
mask_diff = (abs(dk_diff) <= Ar); 
mask_sum  = (abs(dk_sum)  <= Ar);

center = A_next + 1;
idx_1 = 1 + A_next + 1;

% Construction of S1 and S2
S_full_blocks = cell(n, n);
S_reduced_blocks = cell(n, n);

for j = 1 : n
    for jp = 1 : n

        % Retrieve the convolution kernel for the current sub-block
        curr_phi = phi_hat{j, jp};
        
        % Construct S1 (shift term k - k') and S2 (sum term k + k') separately
        S1_mat = zeros(L_next, L_next);
        S1_mat(mask_diff) = curr_phi(dk_diff(mask_diff) + center);
        
        S2_mat = zeros(L_next, L_next);
        S2_mat(mask_sum) = curr_phi(dk_sum(mask_sum) + center);
        
        % Merge into full-scale sub-blocks
        S_full_blocks{j, jp} = S1_mat + S2_mat;
        
        % Row and column excision
        S_temp = S_full_blocks{j, jp};

        if j == torus_idx
            S_temp(idx_1, :) = []; 
        end

        if jp == torus_idx
            S_temp(:, idx_1) = []; 
        end
        
        % Store the reduced sub-block
        S_reduced_blocks{j, jp} = S_temp;
    end
end

S = cell2mat(S_reduced_blocks);




% --- Step 3: Construct the frequency derivative part B of T ---
B_full_blocks = cell(n, n);
B_reduced_blocks = cell(n, n);

% Calculate full-scale sub-blocks for B and excise rows and columns for each sub-block
for j = 1 : n
    % Construct row-generating vector V_row
    V_row = -ks_curr .* z_appro_expand(:, j);
    
    for jp = 1 : n
        % Construct column-generating vector V_col 
        V_col = S_full_blocks{torus_idx, jp}(idx_1, :).';
        
        % Construct the B matrix
        B_sub_full = (eps / a) * (V_row * V_col.'); 
        B_full_blocks{j, jp} = B_sub_full;
        
        % Row and column excision
        B_temp = B_sub_full;
        if j == torus_idx,  B_temp(idx_1, :) = []; end
        if jp == torus_idx, B_temp(:, idx_1) = []; end

        % Store the reduced sub-block
        B_reduced_blocks{j, jp} = B_temp;
    end
end

B = cell2mat(B_reduced_blocks);




% --- Step 4: Assemble the complete linearized operator T ---
T = D + eps * S + B;


end