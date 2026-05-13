%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/07  Mingwei Fu                                             %%%
%%% This function expands a Fourier coefficient matrix from a smaller  %%%
%%% scope [-A, A]^2 to a larger scope [-B, B]^2 using zero-padding.    %%%
%%% The center of the matrix is maintained at (0,0).                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input:                                                             %%%
%%% M_small:  The original matrix of size (2*A + 1) x (2*A + 1)        %%%
%%% A:        The original truncation radius                           %%%
%%% B:        The target truncation radius (B > A)                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Output:                                                            %%%
%%% M_large:  The expanded matrix of size (2*B + 1) x (2*B + 1)        %%%
%%%           with M_small at its center and zero-padding elsewhere.   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function M_large = Matrix_Expand_Padding(M_small, A, B)

if B <= A
    error('B must be greater than A');
end

if size(M_small, 1) ~= (2*A + 1)
    error('v_small length does not match parameter A');
end



L_A = 2 * A + 1;
L_B = 2 * B + 1;


M_large = zeros(L_B, L_B);
offset = B - A;
M_large(offset + 1 : offset + L_A, offset + 1 : offset + L_A) = M_small;
    
end