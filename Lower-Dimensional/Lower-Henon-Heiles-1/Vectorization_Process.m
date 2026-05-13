%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/10  Mingwei Fu                                             %%%
%%% This function process rearrangement and deleting of the            %%%
%%% P-equations.                                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% matrix_ori: L_next x n matrix, which is P-equations                %%%
%%%             in [-A^{r+1}, A^{r+1}]                                 %%%
%%% A_next:     current truncation scope A^{r+1}                       %%%
%%% torus_idx:  index of the chosen torus                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% matrix_nan:  P-equations values in forms of matrix of size         %%%
%%%              (Ar*A, n) setting nan at the Q-eqns                   %%%
%%% vec_tot_nan: the whole vectors with nan at Q-eqns                  %%%
%%% vec_reduced: the whole vectors after deleting Q-equation           %%%
%%%              terms                                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [matrix_nan, vec_tot_nan, vec_reduced] = Vectorization_Process(matrix_ori, A_next, torus_idx)


% Step 1: Set the resonance terms in the P-equation tensor to NaN
idx_1  =  1 + A_next + 1;   % k = 1

matrix_nan = matrix_ori;
matrix_nan(idx_1, torus_idx) = nan;


% Step 2: Integrate into a complete global vector containing Q-equation terms
vec_tot_nan = matrix_nan(:);


% Step 3: Obtain the reduced global vector by removing Q-equation terms
vec_reduced = vec_tot_nan(~isnan(vec_tot_nan));  


end