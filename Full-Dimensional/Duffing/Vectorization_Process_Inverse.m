%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This function reverses the vectorization process:                  %%%
%%% It restores a reduced vector back to a full entry vector by        %%%
%%% re-inserting 0s at resonance positions and reshaping.              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% vector_reduced: (L_next-1) vector, which is reduced                %%% 
%%% A_next:         current truncation scope A^{r+1}                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% vector_ori: length L_next vector, which is reinserted 0s at        %%%
%%%             resonance positions in interval [-A^{r+1}, A^{r+1}]    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function vector_ori = Vectorization_Process_Inverse(vector_reduced, A_next)


L_next = 2 * A_next + 1;
idx_1  =  1 + A_next + 1;   

% Create a logical mask initialized to true and set the resonance position to false
mask = true(L_next, 1);
mask(idx_1) = false;

vector_ori = zeros(L_next, 1);
vector_ori(mask) = vector_reduced;

end