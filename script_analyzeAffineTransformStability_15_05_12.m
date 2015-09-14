foldePattern = 'T:\temp\deconvolution\15_05_12_fly_functionalImage\TM??????'
TMvec = [2500:50:3450 3600];%we have removed outliers after analysis

outputFolderPattern = ['T:\temp\deconvolution\15_05_12_fly_functionalImage_cluster\TM??????']

numViews = 4;
medFilterRadius = 5
%%
N = length(TMvec);

Astruct(N,numViews).A = [];%preallocate
for ii = 1:N
    TM = TMvec(ii);
    
    baseRegistrationFolder = recoverFilenameFromPattern(foldePattern,TM);
    
    %load coarse registration
    coarse = load([baseRegistrationFolder filesep  'imRegister_Matlab_tform.mat'],'tformCell');
    
    %load fine registration
    fine = load([baseRegistrationFolder filesep  'imWarp_Matlab_tform_fine.mat'],'tformCell');
    
    for jj = 1:size(Astruct,2)
        A = coarse.tformCell{jj} * fine.tformCell{jj};
        Astruct(ii,jj).A = A;
        %decompose all matrices into basic unites
        [Astruct(ii,jj).R, Astruct(ii,jj).T, Astruct(ii,jj).S] = affineTransformDecomposition(A);
    end
end

%%
%plot different elements (just select different elements)
qq = zeros(N,1);
for jj = 1:numViews
    for ii = 1:N
        qq(ii) = Astruct(ii,jj).T(1);
    end
    figure;plot(TMvec,qq);
    hold on;
    plot(TMvec,medfilt1(qq,medFilterRadius),'r');
    title(['View ' num2str(jj)]);
end


%%
%perform interpolation for each element
numFrames = TMvec(end)-TMvec(1)+1;
Xq = [TMvec(1):TMvec(end)];
TMfinal = Xq;
Afinal(numFrames, numViews).R = zeros(3);
Afinal(numFrames, numViews).S = zeros(3);
Afinal(numFrames, numViews).T = zeros(1,3);
qq = zeros(1,N);
for jj = 1:numViews
    %translation
    for kk = 1:3
        for ii = 1:N
            qq(ii) = Astruct(ii,jj).T(kk);
        end
        qq = medfilt1(qq,medFilterRadius);
        Vi = interp1(TMvec, qq', Xq, 'linear');
        
        for aa = 1:numFrames
            Afinal(aa,jj).T(kk) = Vi(aa);
        end
        
    end
    
    %scale    
    for kk = 1:3
        for mm = 1:3
            for ii = 1:N
                qq(ii) = Astruct(ii,jj).S(kk,mm);
            end
            qq = medfilt1(qq,medFilterRadius);
            Vi = interp1(TMvec, qq', Xq, 'linear');
            
            for aa = 1:numFrames
                Afinal(aa,jj).S(kk,mm) = Vi(aa);
            end
            
        end
    end
    
    %rotation    
    for kk = 1:3
        for mm = 1:3
            for ii = 1:N
                qq(ii) = Astruct(ii,jj).R(kk,mm);
            end
            qq = medfilt1(qq,medFilterRadius);
            Vi = interp1(TMvec, qq', Xq, 'linear');
            
            for aa = 1:numFrames
                Afinal(aa,jj).R(kk,mm) = Vi(aa);
            end
            
        end
    end
    %find the closest rotation matrix
    %TODO: use quaternions for this interpolation
    for aa = 1:numFrames
        R =  Afinal(aa,jj).R(1:3,1:3);
        [R, ~, S] = poldecomp(R);
        Afinal(aa,jj).R = R;
        
        Afinal(aa,jj).S = Afinal(aa,jj).S(1:3,1:3);
        
        %calculate final A
        Afinal(aa,jj).A = eye(4);
        Afinal(aa,jj).A(1:3,1:3) = Afinal(aa,jj).S * Afinal(aa,jj).R;
        Afinal(aa,jj).A(4,1:3) = Afinal(aa,jj).T;
    end
    
    
end

%%
%plot different elements (just select different elements)
qq = zeros(numFrames,1);
for jj = 1:numViews
    for ii = 1:numFrames
        qq(ii) = Afinal(ii,jj).S(1,1);
    end
    figure;plot(TMfinal,qq);    
    title(['View ' num2str(jj) ' final values']);
end

%%
%save results
for ii = 1:numFrames
   outFolder = recoverFilenameFromPattern(outputFolderPattern,TMfinal(ii));
   if( exist(outFolder) == 0 )
       mkdir(outFolder);
   end
   
   for jj = 1:numViews
       fid = fopen([outFolder filesep 'affineTr_view' num2str(jj) '.txt'],'w');
       fprintf(fid,'%.12f %.12f %.12f %.12f\n',Afinal(ii,jj).A');
       fclose(fid);
   end
end

