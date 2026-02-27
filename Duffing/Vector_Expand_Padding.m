%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This function expands a vector from a smaller scope                %%%
%%% [-A, A] to a larger scope [-B, B] using zero-padding.              %%%
%%% The center of the vector is maintained at 0.                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% v_small: The original vector of length (2*A + 1)                   %%%
%%% A:       The original truncation radius                            %%%
%%% B:       The target truncation radius (B > A)                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% v_large: The expanded vector of length (2*B + 1)                   %%%
%%%          with v_small at its center and zero-padding elsewhere.    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function v_large = Vector_Expand_Padding(v_small, A, B)

if B <= A
    error('B must be greater than A');
end

if length(v_small) ~= (2*A + 1)
    error('v_small length does not match parameter A');
end


L_A = 2 * A + 1;
L_B = 2 * B + 1;


v_large = zeros(L_B, 1);
offset  = B - A;
v_large(offset + 1 : offset + L_A) = v_small;
    
end