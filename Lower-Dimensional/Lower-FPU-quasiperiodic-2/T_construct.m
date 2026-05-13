%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/21  Mingwei Fu                                             %%%
%%% This function constructs the linearized operator                   %%%
%%% T_{A^{r+1}}, which is size ( (2A^{r+1}+1)^2 * n - m ) \times       %%%
%%% ( (2A^{r+1}+1)^2 * n - m ).                                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:   tensor of z_1 and z_2 coefficients                      %%%
%%% mu_appro:  approximate frequencies (dim = m = 2)                   %%%
%%% mu:        initial frequencies (dim = n)                           %%%
%%% eps:       perturbation quantity                                   %%%
%%% a:         initial coefficients                                    %%%
%%% n:         dimension of torus                                      %%%
%%% torus_idx: indices of these chosen tori                            %%%
%%% A:         truncation parameter                                    %%%
%%% V:         the coefficient tensor of nonlinear terms               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% T: linearized matrix of size ( (2A^{r+1}+1)^2 * n - m )  ×         %%%
%%%    ( (2A^{r+1}+1)^2 * n - m ), with resonant entries reduced       %%%
%%% D: diagonal part of the linearized matrix                          %%%
%%% S: the linearized matrix of the nonlinear term                     %%%
%%% B: the additional matrix arises from the difference of approximate %%%
%%%    frequencies                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [T, D, S, B] = T_construct(z_appro, mu_appro, mu, eps, a, n, torus_idx, A, V)


% settings
m      = size(torus_idx, 1);
Lr     = size(z_appro(:, :, 1), 1);  % Lr = 2 * A^r + 1
Ar     = (Lr - 1) / 2;               % Ar = A^r
A_next = Ar * A;                     % A_next = A^{r+1}
L_next = 2 * A_next + 1;             % L_next = 2 * A^{r+1} + 1

z_appro_bar = zeros(Lr, Lr, n);
for j = 1 : n
    z_appro_bar(:, :, j) = rot90(z_appro(:, :, j), 2);
end




% --- Step 1: Construct the diagonal part D of operator T ---
ks_curr    = -A_next : A_next; 
[K1, K2]   = ndgrid(ks_curr, ks_curr);  
inner_prod = K1 * mu_appro(1) + K2 * mu_appro(2); 

D_part_tensor = zeros(L_next, L_next, n);  
for j = 1 : n
    D_part_tensor(:, :, j) = -inner_prod + mu(j);
end

[~, ~, ~, D_vec_red_tot] = Vectorization_Process(D_part_tensor, A_next, n, torus_idx);
D = diag(D_vec_red_tot); 




% --- Step 2: Construct the nonlinear perturbation part S of T ---
q = cell(n, 1);  
for j = 1 : n
    q{j, 1} = z_appro(:, :, j) + z_appro_bar(:, :, j);
end

Ar_expand = 2 * Ar;  
Lr_expand = 2 * Ar_expand + 1;


% Calculate the second derivative matrix (Hessian)
H1_partial_Hessian = cell(n, n);

for j1 = 1 : n
    for j2 = 1 : n
        hess_sum = zeros(Lr_expand, Lr_expand);
        for j3 = 1 : n
            for j4 = 1 : n
                if V(j1, j2, j3, j4) == 0, continue; end
                
                hess_sum = hess_sum + 12 * V(j1, j2, j3, j4) * conv2(q{j3}, q{j4});
            end
        end
        H1_partial_Hessian{j1, j2} = hess_sum;
    end
end

% Expand each sub-block of the Hessian matrix to the interval [-A^{r+1}, A^{r+1}]^2
offset_2 = A_next - Ar_expand;
H1_partial_Hessian_next = cell(n, n);

for j1 = 1 : n
    for j2 = 1 : n
        submat_temp = zeros(L_next, L_next);
        submat_temp(offset_2 + 1 : offset_2 + Lr_expand, offset_2 + 1 : offset_2 + Lr_expand) = H1_partial_Hessian{j1, j2};
        H1_partial_Hessian_next{j1, j2} = submat_temp;
    end
end

% Prepare the linear index offset mapping for the 2D frequency grid
ks = (-A_next : A_next)';
[K1_grid, K2_grid] = ndgrid(ks, ks); 
K1_v = K1_grid.';  K1_v = K1_v(:);
K2_v = K2_grid.';  K2_v = K2_v(:);
center = A_next + 1; 

% Calculate frequency displacement matrices (k - k' and k + k')
dK1 = K1_v - K1_v.';  dK2 = K2_v - K2_v.';
sK1 = K1_v + K1_v.';  sK2 = K2_v + K2_v.';

% Identify masks for displacements within the valid support set [-Ar_expand, Ar_expand]^2
mask_d = (abs(dK1) <= Ar_expand) & (abs(dK2) <= Ar_expand);
mask_s = (abs(sK1) <= Ar_expand) & (abs(sK2) <= Ar_expand);

% Locate the indices of resonance points e1=(1,0) and e2=(0,1) in the row-major scanning vector
e_vec = [1, 0; 
         0, 1];
e_idx = zeros(2, 1);
for k = 1:2
    [~, e_idx(k)] = ismember(e_vec(k, :), [K1_v, K2_v], 'rows');
end
idx_e1 = e_idx(1); 
idx_e2 = e_idx(2);

% Construction of S1 and S2
S_full_blocks = cell(n, n);
S_reduced_blocks = cell(n, n);

for j = 1:n
    for jp = 1:n

        % Retrieve the convolution kernel for the current sub-block
        curr_phi = H1_partial_Hessian_next{j, jp};
        
        % Construct S1 (shift term k - k') and S2 (sum term k + k') separately
        S1_mat = zeros(L_next^2, L_next^2);
        idx_d = sub2ind([L_next, L_next], dK1(mask_d) + center, dK2(mask_d) + center);
        S1_mat(mask_d) = curr_phi(idx_d);
        
        S2_mat = zeros(L_next^2, L_next^2);
        idx_s = sub2ind([L_next, L_next], sK1(mask_s) + center, sK2(mask_s) + center);
        S2_mat(mask_s) = curr_phi(idx_s);
        
        % Merge into full-scale sub-blocks
        S_full_blocks{j, jp} = S1_mat + S2_mat;
        
        % Row and column excision
        r_rem = []; c_rem = [];
        if j == torus_idx(1),  r_rem = idx_e1; elseif j == torus_idx(2), r_rem = idx_e2; end
        if jp == torus_idx(1), c_rem = idx_e1; elseif jp == torus_idx(2), c_rem = idx_e2; end
        
        S_temp = S_full_blocks{j, jp};
        if ~isempty(r_rem), S_temp(r_rem, :) = []; end
        if ~isempty(c_rem), S_temp(:, c_rem) = []; end
        
        % Store the reduced sub-block
        S_reduced_blocks{j, jp} = S_temp;
    end
end

S = cell2mat(S_reduced_blocks);




% --- Step 3: Construct the frequency derivative part B of T ---
B_full_blocks = cell(n, n);
for j = 1:n
    for jp = 1:n
        B_full_blocks{j, jp} = zeros(L_next^2, L_next^2); 
    end
end

% Expand z_appro and z_appro_bar to [-A^{r+1}, A^{r+1}]^2 and pad with zeros
offset_1 = A_next - Ar;
z_appro_expand     = zeros(L_next, L_next, n);  

for j = 1 : n
    z_appro_expand(offset_1 + 1 : offset_1 + Lr, offset_1 + 1 : offset_1 + Lr, j) = z_appro(:, :, j);
end

% Sum contributions from the m=2 torus frequencies
for l = 1 : 2 
    kl_v = (l == 1) * K1_v + (l == 2) * K2_v;
    idx_el = e_idx(l); 
    
    for j = 1 : n
        z_curr = z_appro_expand(:, :, j).'; 

        % Construct row-generating vector V_row
        V_row = -kl_v .* z_curr(:); 

        for jp = 1 : n
            % Construct column-generating vector V_col
            V_col = S_full_blocks{torus_idx(l), jp}(idx_el, :).'; 

            % Construct the B matrix
            B_full_blocks{j, jp} = B_full_blocks{j, jp} + (eps / a(l)) * (V_row * V_col.');
        end
    end
end

% Perform block-wise excision for B
B_reduced_blocks = cell(n, n);
for j = 1:n
    for jp = 1:n
        r_rem = []; c_rem = [];
        if j == torus_idx(1), r_rem = idx_e1; elseif j == torus_idx(2), r_rem = idx_e2; end
        if jp == torus_idx(1), c_rem = idx_e1; elseif jp == torus_idx(2), c_rem = idx_e2; end
        
        B_temp = B_full_blocks{j, jp};
        if ~isempty(r_rem), B_temp(r_rem, :) = []; end
        if ~isempty(c_rem), B_temp(:, c_rem) = []; end
        B_reduced_blocks{j, jp} = B_temp;
    end
end
B = cell2mat(B_reduced_blocks);




% --- Step 4: Assemble the complete linearized operator T ---
T = D + eps * S + B;


end