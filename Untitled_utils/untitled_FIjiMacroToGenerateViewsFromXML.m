%function untitled_FIjiMacroToGenerateViewsFromXML()

%nuclei channel
suffixCell = {'', 'membrCh'};
for TM = 939:1100
    for ii = 1:length(suffixCell)
        suffix = suffixCell{ii};
        fid = fopen(['b:\keller\15_06_11_fly_functionalImage\TM' num2str(TM,'%.6d') suffix '_Fiji\Macro.ijm'],'w');
        if( fid <0 )
           display(['CHECK TM ' num2str(TM)])
           continue;
        end
        %images
        fprintf(fid,'run("Fuse/Deconvolve Dataset", "select_xml=/nobackup/keller/15_06_11_fly_functionalImage/TM%.6d%s_Fiji/dataset.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_timepoint=[All Timepoints] type_of_image_fusion=[No fusion, create individual registered images] bounding_box=[Use pre-defined Bounding Box] fused_image=[Save as TIFF stack] bounding_box_title=[My Bounding Box] downsample=1 pixel_type=[32-bit floating point] imglib2_container=ArrayImg interpolation=[Linear Interpolation] output_file_directory=/nobackup/keller/15_06_11_fly_functionalImage/TM%.6d%s_Fiji/.")\n',TM,suffix, TM, suffix);
        %weights: we need to save them in different folders
        outputFolderW = ['b:/keller/15_06_11_fly_functionalImage/TM' num2str(TM,'%.6d') suffix '_Fiji/ww'];
        if( exist(outputFolderW) == 0 )
            mkdir(outputFolderW);
        end
        fprintf(fid,'run("Fuse/Deconvolve Dataset", "select_xml=/nobackup/keller/15_06_11_fly_functionalImage/TM%.6d%s_Fiji/dataset_weights.xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_timepoint=[All Timepoints] type_of_image_fusion=[No fusion, create individual registered images] bounding_box=[Use pre-defined Bounding Box] fused_image=[Save as TIFF stack] bounding_box_title=[My Bounding Box] downsample=1 pixel_type=[32-bit floating point] imglib2_container=ArrayImg interpolation=[Linear Interpolation] output_file_directory=/nobackup/keller/15_06_11_fly_functionalImage/TM%.6d%s_Fiji/ww/.")\n',TM,suffix,  TM, suffix);
        
        fprintf(fid, 'run("Quit")\n');%to exit Fiji in headless mode
        fclose(fid);
    end
end

