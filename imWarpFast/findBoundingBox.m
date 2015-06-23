%ROI : 2x3 array with xyzMin on top row and xyzMax on Bottom row
function ROI = findBoundingBox(A, imSize)

ROI = [inf inf inf; -inf -inf -inf];
for xx = [0 imSize(1)]
    for yy = [0 imSize(2)]
        for zz = [0 imSize(3)]
            p = A * [xx yy zz 1]';
            ROI(1,:) = min(ROI(1,:),p(1:3)');
            ROI(2,:) = max(ROI(2,:),p(1:3)');
        end
    end
end