%filename CSV is a comma separated file where each row contains the following 8 numbers:
%viewA, viewB, x_A, y_A, z_A, x_B, y_B, z_B
%The coordinates correspond to opening files with ImageJ
%The first view is considered view with index 1
%At least 4 points are needed for each pair of views

function Acell = fitMultiviewAffineFromManualLandmarks(filenameCSV)

qq = load(filenameCSV);

%%
%parse information
Nviews = max(max(qq(:,1:2)));

Tcell = cell(Nviews);
countM = zeros(Nviews);
for ii = 1:size(qq,1)
   mm = qq(ii,1);
   nn = qq(ii,2);
   countM(nn,mm) = countM(nn,mm) + 1;
   
   if( countM(nn,mm) == 1)
       Tcell{mm,nn} = cell(1);
   end
   
   xyz = [qq(ii,[4 3 5 7 6 8]) 1.0];%permute x and y and add NCC fake value
   Tcell{qq(ii,1),qq(ii,2)}{countM(nn,mm)} = xyz;
end


%%
%fit RANSAC
maxRadiusResidualInPixels = 15;
[A, stats] = fitAffineMultiviewRANSACiteration(Tcell, maxRadiusResidualInPixels);

Acell = cell(Nviews,1);
Acell{1} = eye(4);
for ii = 2:Nviews
   Acell{ii} = [reshape(A(12 * (ii-2) + 1: 12 *(ii-1)),[4 3]), [0;0;0;1]]; 
end

%%
%display stats
rr = sqrt(sum(stats.residuals.^2,2));
N = sum(countM(:));
display(['Avg. residual = ' num2str(mean(rr)) ' pixels with ' num2str(stats.numInliers) ' inliers out of ' num2str(N) ' pairwise landmarks']);

figure; 
hist(rr,[0:0.5:max(rr)]);xlim([0 max(rr)]);