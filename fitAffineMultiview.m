function tformCell = fitAffineMultiview(Tcell)
%%
confidenceRatio = 1.00;

%%
ii = 1;
jj = 2;

ctrpts = Tcell{ii,jj}; %control points

%select best match only if weights is above confidence ratio
X = zeros(10000,3);
Y = X;
W = zeros(size(X,1),1);
nn = 0;

for kk = 1:length(ctrpts)
   if( isempty(ctrpts{kk}) )
       continue;
   end
   
   ctrpts{kk}(:,7)
   if( ctrpts{kk}(1,7) > confidenceRatio * ctrpts{kk}(2,7) )
      nn = nn + 1;
      X(nn,:) = ctrpts{kk}(1,1:3);
      Y(nn,:) = ctrpts{kk}(1,4:6);
      W(nn) = ctrpts{kk}(1,7);
   end
end

X(nn + 1:end,:) = [];
Y(nn + 1:end,:) = [];
W(nn + 1:end) = [];

%figure;plot3(X(:,1),X(:,2), X(:,3),'o');grid on; hold on;plot3(Y(:,1),Y(:,2), Y(:,3),'r+');

%%
%rigid transformation (plus scaling)
[d,Z,transform] = procrustes(X,Y, 'reflection', false);%Z = b*Y*T + c;

%%
%generate matrices to fit multi-view affine using least squares
X = [X ones(size(X,1),1)];
H = sparse(size(X,1) * 3, size(X,2) * 3);
for ii = 1:3
    H((ii-1) * size(X,1) + 1: ii * size(X,1),(ii-1) * size(X,2) + 1: ii * size(X,2)) = X;
end

Y = Y(:);

A = H \ Y;

rr = H * A - Y;