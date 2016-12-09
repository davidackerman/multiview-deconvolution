function [PSF,erase] = trim_PSF(PSF, thr)

if( nargin < 2 )
    thr = 1e-8;
end

BW = PSF > thr ;

n_dims = ndims(BW) ;

erase = zeros(n_dims,1);
for ii = 1:n_dims
   
    %reduce all the dimensions that are not of interest
    qq = BW;
    for jj = n_dims:-1:1
       if( ii == jj )
           continue;
       end
       qq = sum(qq,jj);
    end
    qq = squeeze(qq);
    
    %figure out how much we can cut: the triming has to be symmetric to
    %keep PSF centered
    p1 = find(qq>0,1,'first') - 2;
    if isempty(p1) ,
        % In this case, we pretend the first element was superthreshold to prevent trimming away to nothingness
        p1 = 1 ;
    else
        p1 = max(1,p1);
    end
    p2 = find(qq>0,1,'last') + 2;
    if isempty(p2) ,
        % In this case, we pretend the last element was superthreshold to prevent trimming away to nothingness
        p2 = length(qq) ;
    else
        p2 = min(length(qq), p2);
    end       
    
    erase(ii) = min(p1, length(qq(p2:end)) ); 
    
end

if n_dims==2 ,
    PSF = PSF(erase(1):end-erase(1)+1, erase(2):end-erase(2)+1);
elseif n_dims==3 ,
    PSF = PSF(erase(1):end-erase(1)+1, erase(2):end-erase(2)+1, erase(3):end-erase(3)+1);
else
    error('Unsupported arity');
end
