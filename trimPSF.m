function PSF = trimPSF(PSF, thr)

if( nargin < 2 )
    thr = 1e-8;
end

BW = PSF > thr;

erase = zeros(3,1);
for ii = 1:ndims(BW)
   
    %reduce all the dimensions that are not of interest
    qq = BW;
    for jj = ndims(BW):-1:1
       if( ii == jj )
           continue;
       end
       qq = sum(qq,jj);
    end
    qq = squeeze(qq);
    
    %figure out how much we can cut: the triming has to be symmetric to
    %keep PSF centered
    p1 = find(qq>0,1,'first') - 2;
    p1 = max(1,p1);
    p2 = find(qq>0,1,'last') + 2;
    p2 = min(length(qq), p2);
    
    erase(ii) = min(p1, length(qq(p2:end)) ); 
    
end

PSF = PSF(erase(1):end-erase(1)+1, erase(2):end-erase(2)+1, erase(3):end-erase(3)+1);