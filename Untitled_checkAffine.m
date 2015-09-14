D = dir('S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\MVrefine_deconv_LR_multiGPU_param_TM*');
for ii = 1:length(D)
    Acell = readXMLdeconvolutionFile(['S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\' D(ii).name]);
    if( checkAffineTr(Acell) == false )
        disp(['File ' D(ii).name ' is wrong']);
    end
end

%%
D = dir('S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\MVrefine_deconv_LR_multiGPU_param_TM*');
G = zeros(4,4,length(D));
for ii = 1:length(D)
    Acell = readXMLdeconvolutionFile(['S:\SiMView1\15-04-03\Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked\' D(ii).name]);
    if( checkAffineTr(Acell) == false )
        disp(['File ' D(ii).name ' is wrong']);
    end
    G(:,:,ii) = Acell{2};
end
