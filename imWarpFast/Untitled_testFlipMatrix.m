%random affine matrix
A = [[rand(3,3); [ 0 0 0 ] ] , [rand(3,1); 1]]
%%
%flip x and y
Af = A;
%Af(1:2,:) = Af([2 1],:);
%Af(:,1:2) = Af(:,[2 1]);
%Af(1:2,1:2) = Af(1:2,1:2)';

Af = [A(2,2) A(2,1) A(2,3) A(2,4);...
      A(1,2) A(1,1) A(1,3) A(1,4);...
      A(3,2) A(3,1) A(3,3) A(3,4);...
      0     0       0       1]
    

%%
%test results
xyz = [1 0 0 1]';
A*xyz
qq= Af*xyz([2 1 3 4]);
qq([2 1 3 4])


%test results
xyz = [rand(3,1); 1];
A*xyz
qq= Af*xyz([2 1 3 4]);
qq([2 1 3 4])