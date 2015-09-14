%transformation is S*R + T
function [R, T, S] = affineTransformDecomposition(A)

T = A(4,1:3);
[R, ~, S] = poldecomp(A(1:3,1:3));