%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This function process deleting of the P-equations.                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% matrix_ori: L_next rows vector, which is P-equations               %%%
%%%             in [-A^{r+1}, A^{r+1}]                                 %%%
%%% A_next:     current truncation scope A^{r+1}                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% vector_nan:     P-eqns vector by setting nan at Q-eqn terms        %%%
%%% vector_reduced: P-eqns vector by deleting Q-eqn terms              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [vector_nan, vector_reduced] = Vectorization_Process(vector_ori, A_next)


idx_1  =  1 + A_next + 1;  % Locate the index of the resonance term in the vector

vector_nan = vector_ori;
vector_nan(idx_1, 1) = nan;
vector_reduced = vector_nan(~isnan(vector_nan));


end