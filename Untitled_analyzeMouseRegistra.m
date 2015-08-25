%check at the refinement coefficients from independent mouse alignment
TMvec = [191:270];


%%

Acell = cell(4,length(TMvec));
count = 0;
for TM = TMvec
    count = count + 1;
    qq = load(['S:\SiMView1\15-08-10\Mmu_E1_mKate2_20150810_160708.corrected\MVrefine_reg_workspace_TM000' num2str(TM)]);
    Acell(:,count) = qq.tformFineCell;
end

%%
%plot different parameters
for vv = 2:4
    view_ = vv;
    
    ll = zeros(size(Acell,2),3);
    
    for kk = 1:size(ll,2)
        coeff = [kk,kk];
        for ii = 1:length(ll)
            ll(ii,kk) = Acell{view_, ii}(coeff(1),coeff(2));
        end
    end
    
    %it oscilates around 1
    (sum(ll) - length(ll))
    
    figure;
    plot(TMvec, ll);
    xlabel('Time point');
    title(['Mouse registration']);
    ylabel(['View ' num2str(view_) ' refined coeff']);
    ylim([0.99 1.01]);
end


%%
%plot different parameters
for vv = 2:4
    view_ = vv;
    
    ll = zeros(size(Acell,2),3);
    
    for kk = 1:size(ll,2)
        coeff = [4,kk];
        for ii = 1:length(ll)
            ll(ii,kk) = Acell{view_, ii}(coeff(1),coeff(2));
        end
    end
    
    %it oscilates around 1
    sum(ll)
    
    figure;
    plot(TMvec, ll);
    xlabel('Time point');
    title(['Mouse registration. Translation parameters']);
    ylabel(['View ' num2str(view_) ' refined coeff']);    
end