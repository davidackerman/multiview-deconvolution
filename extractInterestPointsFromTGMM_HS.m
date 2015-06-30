% % Run these two comand to generate segmentation mask
% % 
% % >ProcessStack.exe T:/temp/deconvolution/15_06_11_fly_functionalImage_TM000001_Fiji/simview3_view3 2 2 180 74
% % >ProcessStack.exe T:/temp/deconvolution/15_06_11_fly_functionalImage_TM000001_Fiji/simview3_view3_hierarchicalSegmentation_conn3D74_medFilRad2.bin 12 50

function extractInterestPointsFromTGMM_HS(filePattern, maxSize, imBackground)


if(nargin < 1 )
    filePattern = 'T:\temp\deconvolution\15_06_11_fly_functionalImage_TM000001_Fiji_TGMM\simview3_view?'
    maxSize = 2500;
    imBackground = 100;
end

[pathstr,name,ext] = fileparts( filePattern );

outputFolder = [pathstr filesep 'interestpoints'];
if( exist(outputFolder) == 0 )
    mkdir(outputFolder);
end

outputPattern = [outputFolder filesep 'tpId_0_viewSetupId_?.beadsTGMM.ip.txt'];

for view = 1:4
    
   filename = recoverFilenameFromPattern(filePattern,view-1);
   im = readTIFFstack([filename '.tif']) - imBackground;
   im = single(im);
   imHS = readTIFFstack([filename '_hierarchicalSegmentation_conn3D74_medFilRad2.bin_tau12.tif']);
   
   CC = regionprops(imHS, 'PixelIdxList','Area');
   
   N = length(CC);
   xyz = zeros(N,4);
   xyz(:,1) = [0:N-1]';
   %filter
   count = 0;
   for ii = 1:N
       if(CC(ii).Area >= maxSize)
           continue;
       end
       
       count = count + 1;
       [x,y,z] = ind2sub(size(im),CC(ii).PixelIdxList);
       ww = im(CC(ii).PixelIdxList).^2;
       ww = ww / sum(ww);
       xyz(count,2:4) = mean(bsxfun(@times,[x y z], ww));%weighted average
   end
   xyz = xyz(1:count,:) - 1;%TO C-indexing
   
   fileout = recoverFilenameFromPattern(outputPattern,view-1);
   fid = fopen(fileout,'w');
   fprintf(fid, 'id\t x\t y\t z\b');
   fprintf(fid, '%d %f %f %f\n',xyz');
   fclose(fid);
   
end