%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/07  Mingwei Fu                                             %%%
%%% This function reverses the vectorization process:                  %%%
%%% It restores a reduced vector back to a n-layer tensor by           %%%
%%% re-inserting 0s at resonance positions and reshaping.              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% vec_reduced: (L_next-1)^2 x n vector, which is reduced             %%% 
%%% A_next:      current truncation scope A^{r+1}                      %%%
%%% n:           dimension of torus                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% tensor_ori: L_next x L_next x n tensor, which is reinserted 0s at  %%%
%%%             resonance positions and reshaping in                   %%%
%%%             [-A^{r+1}, A^{r+1}]^2                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function tensor_ori = Vectorization_Process_Inverse(vec_reduced, A_next, n)

L_next = 2 * A_next + 1;




% --- Step 1: Create a logical mask in matrix form ---
mask_tensor = true(L_next, L_next, n);

e1 = [1, 0];
e2 = [0, 1];
res_idx = [e1; e2];

for j = 1 : n
    % row = k1 + A + 1, col = k2 + A + 1
    row_idx = res_idx(j, 1) + A_next + 1;
    col_idx = res_idx(j, 2) + A_next + 1;
    
    % Mark resonance positions as false in the mask tensor
    mask_tensor(row_idx, col_idx, j) = false;
end




% --- Step 2: Rearrange the mask matrix into a vector ---
mask_comp_vec = zeros(L_next^2, n);
for j = 1 : n
    M_mask = mask_tensor(:, :, j);
    M_mask_trans = M_mask.'; 
    mask_comp_vec(:, j) = M_mask_trans(:);
end

% Integrate into a total global logical mask
mask_tot = logical(mask_comp_vec(:));




% --- Step 3: Global zero-padding and filling ---
vec_full = zeros(n * L_next^2, 1);

% Use the mask to fill the reduced vector back into non-resonant positions
vec_full(mask_tot) = vec_reduced;




% --- Step 4: Restore to the tensor structure ---
comp_full = reshape(vec_full, [L_next^2, n]);

tensor_ori = zeros(L_next, L_next, n);
for j = 1 : n
    M_trans = reshape(comp_full(:, j), [L_next, L_next]);
    tensor_ori(:, :, j) = M_trans.'; 
end


end