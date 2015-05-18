function debug_checkBlockwiseMatches(imRef, im, Tcell)


%for ii = 1:size(Tcell,1)
for ii = 1:500
    if( isempty(Tcell{ii}) )
        continue;
    end
    
    aux = Tcell{ii};    
    zRef = round(aux(3));
    figure;
    subplot(1,2,1); 
    imagesc(imRef(:,:,zRef)); 
    colormap gray;
    hold on;
    plot(aux(1), aux(2),'ro');
    
    subplot(1,2,2); 
    z = round(aux(6));
    imagesc(im(:,:,z));
    colormap gray;
    hold on;
    plot(aux(4), aux(5),'g+');       
    
    a = 1;
end