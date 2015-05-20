imOrig = zeros(32,48,40);
anisotropy = 12.8;

xyz = randi(20,[20 3]) + 6;
nn = size(xyz,1);

angles = [90:90:270];



for jj = 1:length(angles)

    Tcell = cell(2);
    
    movingPoints = xyz;
    fixedPoints = xyz;
    Tcell{1,2} = cell(nn,1);
    
    for ii = 1:size(xyz,1)
      im = imOrig;
      pp = xyz(ii,:);
      im(pp(1), pp(2), pp(3)) = 10;
      
      [A, im] = coarseRegistrationBasedOnMicGeometry(im,angles(jj), anisotropyZ);
            
      [~,pos] = max(im(:));
      [x,y,z] = ind2sub(size(im),pos);
      movingPoints(ii,:) = [x,y,z];
      
            
      Tcell{1,2}{ii} = [movingPoints(ii,[2 1 3]),fixedPoints(ii,[2 1 3]), 1.0];
            
    end

    [A, stats] = fitAffineMultiviewRANSACiteration(Tcell, 10);
    A = [reshape(A,[4 3]) [0;0;0;1]];
    A = inv(A)
    size(im)
    
    %tform = fitgeotrans(movingPoints,xyz,'affine');
    %[d,Z,transform] = procrustes(fixedPoints,movingPoints, 'reflection', true);%Z = b*movingPoints*T + c;
end
