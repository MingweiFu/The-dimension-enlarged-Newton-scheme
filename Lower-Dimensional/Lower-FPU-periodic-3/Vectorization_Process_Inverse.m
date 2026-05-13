%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/15  Mingwei Fu                                             %%%
%%% This function reverses the vectorization process:                  %%%
%%% It restores a reduced vector back to a n-layer matrix by           %%%
%%% re-inserting 0s at resonance positions and reshaping.              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% vec_reduced: L_next x n - m vector, which is reduced               %%% 
%%% A_next:      current truncation scope A^{r+1}                      %%%
%%% n:           dimension of torus                                    %%%
%%% torus_idx:   index of the chosen torus                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% matrix_ori: L_next x n matrix, which is reinserted 0s at           %%%
%%%             resonance positions and reshaping in                   %%%
%%%             [-A^{r+1}, A^{r+1}]                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function matrix_ori = Vectorization_Process_Inverse(vec_reduced, A_next, n, torus_idx)

L_next = 2 * A_next + 1;




% --- Step 1: Create a logical mask in matrix form ---
idx_1  =  1 + A_next + 1;   % k = 1

mask = true(L_next, n);
mask(idx_1, torus_idx) = false;




% --- Step 2: Rearrange the mask matrix into a vector ---
mask_vec = mask(:);




% --- Step 3: Global zero-padding and filling ---
vec_full = zeros(n * L_next, 1);
vec_full(mask_vec) = vec_reduced;

matrix_ori = reshape(vec_full, [L_next, n]); 





end