%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/07  Mingwei Fu                                             %%%
%%% This function constructs the linearized operator                   %%%
%%% T_{A^{r+1}}, which is size ( (2A^{r+1}+1)^2 ) * n - n \times       %%%
%%% ( (2A^{r+1}+1)^2 ) * n - n.                                        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% z_appro:  tensor of z_1 and z_2 coefficients                       %%%
%%% mu_appro: approximate frequencies                                  %%%
%%% mu:       initial frequencies                                      %%%
%%% eps:      perturbation quantity                                    %%%
%%% a:        initial coefficients                                     %%%
%%% n:        dimension of torus                                       %%%
%%% A:        truncation parameter                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% T: linearized matrix of size ( (2A^{r+1}+1)^2 - 1 ) * n ×          %%%
%%%    ( (2A^{r+1}+1)^2 - 1 ) * n, with resonant entries reduced       %%%
%%% D: diagonal part of the linearized matrix                          %%%
%%% S: the linearized matrix of the nonlinear term                     %%%
%%% B: the additional matrix arises from the difference of approximate %%%
%%%    frequencies                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [T, D, S, B] = T_construct(z_appro, mu_appro, mu, eps, a, n, A)


% settings
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
D_part_tensor(:, :, 1) = -inner_prod + mu(1);
D_part_tensor(:, :, 2) = -inner_prod + mu(2);

[~, ~, ~, D_vec_red] = Vectorization_Process(D_part_tensor, A_next, n);  
D = diag(D_vec_red);  


% Expand z_appro and z_appro_bar to [-A^{r+1}, A^{r+1}]^2 and pad with zeros
offset_1 = A_next - Ar;

z_appro_expand     = zeros(L_next, L_next, n);  
z_appro_bar_expand = zeros(L_next, L_next, n); 

for j = 1 : n
    z_appro_expand(offset_1 + 1 : offset_1 + Lr, offset_1 + 1 : offset_1 + Lr, j)     = z_appro(:, :, j);
    z_appro_bar_expand(offset_1 + 1 : offset_1 + Lr, offset_1 + 1 : offset_1 + Lr, j) = z_appro_bar(:, :, j);
end




% --- Step 2: Construct the nonlinear perturbation part S of T ---
q1_expand = z_appro_expand(:, :, 1) + z_appro_bar_expand(:, :, 1);  % q_1(k) = z_1(k) + \bar{z}_1(k)
q2_expand = z_appro_expand(:, :, 2) + z_appro_bar_expand(:, :, 2);  % q_2(k) = z_2(k) + \bar{z}_2(k)
center = A_next + 1;                                                % Center index for frequency (0,0) in the L_next x L_next matrix

% The second derivative coefficients phi_hat
phi_hat = {(1/sqrt(2))*q2_expand, (1/sqrt(2))*q1_expand; ...
           (1/sqrt(2))*q1_expand, (-1/sqrt(2))*q2_expand};

% Prepare the linear index offset mapping for the 2D frequency grid
ks = (-A_next : A_next)';
[K1_grid, K2_grid] = ndgrid(ks, ks); 
K1_v = K1_grid.'; K1_v = K1_v(:);
K2_v = K2_grid.'; K2_v = K2_v(:);

% Calculate frequency displacement matrices (k - k' and k + k')
dK1 = K1_v - K1_v.';  dK2 = K2_v - K2_v.';
sK1 = K1_v + K1_v.';  sK2 = K2_v + K2_v.';

% Identify masks for displacements within the valid support set [-Ar, Ar]^2
mask_d = (abs(dK1) <= Ar) & (abs(dK2) <= Ar);
mask_s = (abs(sK1) <= Ar) & (abs(sK2) <= Ar);

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
        curr_phi = phi_hat{j, jp};
        
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
        if j == 1,  r_rem = idx_e1; else, r_rem = idx_e2; end
        if jp == 1, c_rem = idx_e1; else, c_rem = idx_e2; end
        
        S_temp = S_full_blocks{j, jp};
        S_temp(r_rem, :) = []; 
        S_temp(:, c_rem) = []; 
        
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


% Calculate full-scale sub-blocks for B (summing contributions from l=1 and l=2)
for l = 1 : n

    % Retrieve the grid vector kl (k1 or k2) for the current frequency dimension
    if l == 1
        kl_v = K1_v; 
    else
        kl_v = K2_v; 
    end
    
    % Retrieve the linear index for the resonance frequency el
    idx_el = e_idx(l); 
    
    for j = 1 : n
        % Construct row-generating vector V_row (associated with component j and dimension l)
        z_curr = z_appro_expand(:, :, j).'; 
        z_v = z_curr(:);
        V_row = -kl_v .* z_v; 
        
        for jp = 1 : n
            % Construct column-generating vector V_col (extracted from the resonance row of S_full_blocks)
            V_col = S_full_blocks{l, jp}(idx_el, :).'; 
            
            % Construct the Bl matrix
            B_full_blocks{j, jp} = B_full_blocks{j, jp} + (eps / a(l)) * (V_row * V_col.');
        end
    end
end


% Excise rows and columns for each sub-block
B_reduced_blocks = cell(n, n);
for j = 1:n
    for jp = 1:n
        % Row and column excision
        if j == 1,  r_rem = idx_e1; else, r_rem = idx_e2; end
        if jp == 1, c_rem = idx_e1; else, c_rem = idx_e2; end
        
        B_temp = B_full_blocks{j, jp};
        B_temp(r_rem, :) = []; 
        B_temp(:, c_rem) = []; 
        
        % Store the reduced sub-block
        B_reduced_blocks{j, jp} = B_temp;
    end
end

B = cell2mat(B_reduced_blocks);




% --- Step 4: Assemble the complete linearized operator T ---
T = D + eps * S + B;


end