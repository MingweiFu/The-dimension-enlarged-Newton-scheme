%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/07  Mingwei Fu                                             %%%
%%% This function process rearrangement and deleting of the            %%%
%%% P-equations.                                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% tensor_ori: L_next x L_next x n tensor, which is P-equations       %%%
%%%             in [-A^{r+1}, A^{r+1}]^2                               %%%
%%% A_next:     current truncation scope A^{r+1}                       %%%
%%% n:          dimension of torus                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% tensor_nan:    P-equations values in forms of tensor of size       %%%
%%%                (Ar*A, Ar*A, n) setting nan at the Q-eqns           %%%
%%% comp_full_nan: n vectors rearranged with nan at Q-eqns             %%%
%%% vec_tot_nan:   the whole vectors with nan at Q-eqns                %%%
%%% vec_reduced:   the whole vectors after deleting Q-equation         %%%
%%%                terms                                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [tensor_nan, comp_full_nan, vec_tot_nan, vec_reduced] = Vectorization_Process(tensor_ori, A_next, n)


L_next = 2 * A_next + 1;


% Step 1: Set the resonance terms in the P-equation tensor to NaN
e1  = [1, 0];
e2  = [0, 1];
res_idx = [e1; e2];

tensor_nan = tensor_ori;
for j = 1 : n
    tensor_nan(res_idx(j, 1) + A_next + 1, res_idx(j, 2) + A_next + 1, j) = nan;
end


% Step 2: Perform full vectorization including Q-equation terms
comp_full_nan = zeros(L_next^2, n);

for j = 1 : n
    M = tensor_nan(:, :, j);
    M_trans = M';
    comp_full_nan(:, j) = M_trans(:);
end


% Step 3: Integrate into a complete global vector containing Q-equation terms
vec_tot_nan = comp_full_nan(:);


% Step 4: Obtain the reduced global vector by removing Q-equation terms
vec_reduced = vec_tot_nan(~isnan(vec_tot_nan));


end