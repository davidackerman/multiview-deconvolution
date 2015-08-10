function bb = checkAffineTr(Acell)

bb = true;
for ii = 1:length(Acell)
    qq = Acell{ii}(1:3,1:3);
    if( abs(det(qq)) < 0.1 ) 
        bb = false;
        break;
    end
end