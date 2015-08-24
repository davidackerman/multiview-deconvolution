function [A, stats] = fitAffineMultiviewRANSACiteration(Tcell, maxRadiusResidualInPixels)

warning off;
numViews = size(Tcell,2);

%4 3D points to fit affine and we have numViews * (numViews - 1) pairs.
Hransac = zeros(4 * 3 * numViews * (numViews - 1), 12 * (numViews-1));%the first view is fixed, so not unknown parameters
Bransac = zeros(4 * 3 * numViews * (numViews - 1),1);

%matrix to hold all the points (full system)
Hall = zeros(10000, 12 * (numViews-1));%the first view is fixed, so not unknown parameters
Ball = zeros(10000, 1);
Nobs = 0;
blockOffset = zeros(size(Tcell,1) * size(Tcell,2),1);%to be able to calculate inliners

countV = 0;
for ii = 1:size(Tcell,1)
    for jj = 1:size(Tcell,2)
        if( isempty(Tcell{ii,jj}) )
            continue;
        end
        countV = countV + 1;
        ctrpts = Tcell{ii,jj}; %control points
        
        %select best match only if weights is above confidence ratio
        X = zeros(length(ctrpts),3);
        Y = X;
        W = zeros(size(X,1),1);
        
        nn = 0;
        
        for kk = 1:length(ctrpts)
            if( isempty(ctrpts{kk}) )
                continue;
            end
            
            %sample according to weighting score
            weightsRnd = ctrpts{kk}(:,7);
            if( sum(weightsRnd) <= 0 )
                continue;
            end
            idx = discreternd(weightsRnd );                        
            nn = nn + 1;
            X(nn,:) = ctrpts{kk}(idx,1:3);
            Y(nn,:) = ctrpts{kk}(idx,4:6);
            W(nn) = ctrpts{kk}(idx,7);
            
        end
        
        if( nn < 4 )%we need at least 4 points
            continue;
        end
        
        if( nn < size(X,1) )
            X(nn + 1:end,:) = [];
            Y(nn + 1:end,:) = [];
            W(nn + 1:end) = [];
        end
        %figure;plot3(X(:,1),X(:,2), X(:,3),'o');grid on; hold on;plot3(Y(:,1),Y(:,2), Y(:,3),'r+');
        
        %subsample solution: select 4 points for RANSAC
        idx = randperm(nn);
        idx = idx(1:4);
        Xransac = X(idx,:);
        Yransac = Y(idx,:);
        Wransac = W(idx);
        
        
        if( jj == 1 )%jj is reference view (no need for H)                          
            Xransac = Xransac(:);
            Bransac((countV-1) * 12 + 1: countV * 12) = Xransac;
        else
            Xransac = [Xransac ones(size(Xransac,1),1)];
            Wx = repmat(Wransac, [1, 4]);
            H = zeros(size(Xransac,1) * 3, size(Xransac,2) * 3);
            Wh = zeros(size(Xransac,1) * 3, size(Xransac,2) * 3);
            for kk = 1:3
                H((kk-1) * size(Xransac,1) + 1: kk * size(Xransac,1),(kk-1) * size(Xransac,2) + 1: kk * size(Xransac,2)) = Xransac;
                Wh((kk-1) * size(Xransac,1) + 1: kk * size(Xransac,1),(kk-1) * size(Xransac,2) + 1: kk * size(Xransac,2)) = Wx;
            end
            Hransac((countV-1) * 12 + 1: countV * 12,(jj-2) * 12 + 1:(jj-1) * 12) = H; %I can add weights if needed
        end
        
        
        if( ii == 1 ) %reference view
            Yransac = Yransac(:);
            Wy = repmat(Wransac, [1, 3]);
            Wy = Wy(:);
            Bransac((countV-1) * 12 + 1: countV * 12) = Yransac;
        else
            
            %Bransac((countV-1) * 12 + 1: countV * 12) = 0;            
                        
            %add Y as a constrain
            Yransac = [Yransac ones(size(Yransac,1),1)];
            H = zeros(size(Yransac,1) * 3, size(Yransac,2) * 3);
            for kk = 1:3
                H((kk-1) * size(Yransac,1) + 1: kk * size(Yransac,1),(kk-1) * size(Yransac,2) + 1: kk * size(Yransac,2)) = Yransac;
            end
            
            if( jj == 1 )
                ss = 1;
            else
                ss = -1;
            end
            Hransac((countV-1) * 12 + 1: countV * 12,(ii-2) * 12 + 1:(ii-1) * 12) = ss * H;
        end
        
        %generate matrices to fit multi-view affine using least squares
        nn = numel(Y);
        if( jj == 1) %referenc view            
            X = X(:);            
            Ball(Nobs +1:Nobs+ nn) = X;%Y .* Wy;
        else
            X = [X ones(size(X,1),1)];
            Wx = repmat(W, [1, 4]);
            H = zeros(size(X,1) * 3, size(X,2) * 3);
            Wh = zeros(size(X,1) * 3, size(X,2) * 3);
            for kk = 1:3
                H((kk-1) * size(X,1) + 1: kk * size(X,1),(kk-1) * size(X,2) + 1: kk * size(X,2)) = X;
                Wh((kk-1) * size(X,1) + 1: kk * size(X,1),(kk-1) * size(X,2) + 1: kk * size(X,2)) = Wx;
            end
            
                        
            Hall(Nobs +1:Nobs+ nn, (jj-2) * 12 + 1:(jj-1) * 12) = H;%right now without weights H .* Wh;
        end
        
        if( ii == 1) %reference view
            Y = Y(:);
            Wy = repmat(W, [1, 3]);
            Wy = Wy(:);
            
            %solution for this particular pair is A = (Wh,*H) \ (Wy .* Y);
            Ball(Nobs +1:Nobs+ nn) = Y;%Y .* Wy;
            
        else
            %Ball(Nobs +1:Nobs+ nn) = 0;
            Y = [Y ones(size(Y,1),1)];
            H = zeros(size(Y,1) * 3, size(Y,2) * 3);
            for kk = 1:3
                H((kk-1) * size(Y,1) + 1: kk * size(Y,1),(kk-1) * size(Y,2) + 1: kk * size(Y,2)) = Y;
            end
            if( jj == 1 )
                ss = 1;
            else
                ss = -1;
            end
            Hall(Nobs +1:Nobs+ nn, (ii-2) * 12 + 1:(ii-1) * 12) = ss * H;
        end
        Nobs = Nobs + nn;
        blockOffset(countV + 1) = Nobs;
       
    end        
end

%"clean" matrices
if( Nobs < size(Ball,1) )
   Hall(Nobs+1:end,:) = [];
   Ball(Nobs+1:end) = [];
end
blockOffset(countV + 2:end) = [];
Hransac = sparse(Hransac);
Hall = sparse(Hall);

%%
%find RANSAC solution
Aransac = Hransac \ Bransac;

%find inliers
ppCell = cell(length(blockOffset)-1,1);
for ii = 1:length(blockOffset)-1
    Haux = Hall(blockOffset(ii)+1:blockOffset(ii+1),:);
    Baux = Ball(blockOffset(ii)+1:blockOffset(ii+1));
    step = size(Haux,1) / 3;%order is [x_1, x_2,...,y_1, y_2, 
    rr = (Haux * Aransac - Baux).^2;
    dd = rr(1:step) + rr(step + 1: 2* step) + rr(2* step + 1:end);
    ppCell{ii} = find( dd < maxRadiusResidualInPixels );
end


stats.maxRadiusResidualInPixels = maxRadiusResidualInPixels;
stats.numInliers = sum(cellfun(@length,ppCell));

%final fit with all the inliers
Hfinal = [];
Bfinal = [];
blockOffsetFinal = zeros(length(ppCell)+1,1);
for ii = 1:length(ppCell)
    Haux = Hall(blockOffset(ii)+1:blockOffset(ii+1),:);
    Baux = Ball(blockOffset(ii)+1:blockOffset(ii+1));
    step = size(Haux,1) / 3;%order is [x_1, x_2,...,y_1, y_2,
    pp = ppCell{ii};
    Hfinal = [Hfinal; [Haux(pp,:); Haux(step + pp,:); Haux(2*step + pp, :)] ];
    Bfinal = [Bfinal; [Baux(pp,:); Baux(step + pp,:); Baux(2*step + pp, :)] ];    
    blockOffsetFinal(ii+1) = size(Hfinal,1);
end
Hfinal = sparse(Hfinal);
A = Hfinal \ Bfinal;

%calculate residuals
stats.residuals = [];
for ii = 1:length(blockOffsetFinal)-1
    Haux = Hfinal(blockOffsetFinal(ii)+1:blockOffsetFinal(ii+1),:);
    Baux = Bfinal(blockOffsetFinal(ii)+1:blockOffsetFinal(ii+1));
    step = size(Haux,1) / 3;%order is [x_1, x_2,...,y_1, y_2, 
    rr = (Haux * A - Baux).^2;    
    stats.residuals = [stats.residuals; [rr(1:step) rr(step + 1: 2* step)  rr(2* step + 1:end)] ];
end

stats.H = Hfinal;
stats.B = Bfinal;
stats.blockOffsetFinal = blockOffsetFinal;



warning on;
%%
%rigid transformation (plus scaling)
%[d,Z,transform] = procrustes(X,Y, 'reflection', false);%Z = b*Y*T + c;

