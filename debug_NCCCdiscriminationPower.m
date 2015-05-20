%function debug_NCCCdiscriminationPower(Tcell)

topN = 3;
NCCscores = zeros(10000,topN);


count = 0;
for ii = 1:size(Tcell,1)
    for jj = 1:size(Tcell,1)
        
        for kk = 1:length(Tcell{ii,jj})
           count = count + 1;
           NCCscores(count,:) = Tcell{ii,jj}{kk}(1:topN,7)';
        end
    end
end

if( count < size(NCCscores,1) )
   NCCscores(count + 1:end,:) = []; 
end

%%
%draw useful plots
figure; 
hist(NCCscores(:,1),[0:0.02:1]);
title('Top score');


figure; 
hist(NCCscores(:,1) ./ NCCscores(:,2),50);
title('Ratio of top 2 score');

figure; 
hist(NCCscores(:,1) ./ NCCscores(:,3),50);
title('Ratio of top 3 score');
