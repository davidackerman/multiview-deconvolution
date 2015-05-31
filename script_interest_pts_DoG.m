%function script_interest_pts_DoG

imPath = 'S:\SiMView3\15-05-22\Dme_E1_57C10_GCaMP6s_Simultaneous_20150522_155301.corrected\SPM00\TM000113\Fiji-deconvolution'

%imgFilepattern = '500um-Beads_SPM00_TM000000_CM0?_CHN00.rotated.unfiltered.raw'
imgFilepattern = 'dataset.h5';
sigma = 2.0;

thrDoG = 20;
minSize = 64;
maxSize = 343;

thrMask = 110;%set to zero to avoid any effect
minIntensityValue = 150;%set to zero to avoid any effect

%%
hinfo = hdf5info([imPath filesep 'dataset.h5']);
if( exist([imPath filesep 'interestpoints']) == 0)
    mkdir([imPath filesep 'interestpoints']);
end

for ii = 0:3
   
   im = hdf5read( hinfo.GroupHierarchy.Groups(6).Groups(ii+1).Groups(1).Datasets(1) );   
   imDoG = imgaussian(single(im),sigma) - imgaussian(single(im),sigma * 1.6);
   
   %mask also areas with low intensity
   mask = im > minIntensityValue;
   if( thrMask > 0 )
       %generate mask (to get features only outside the embryo)
       mask = mask & (~maskEmbryo(im, thrMask));             
   end   
   imDoG = imDoG .* mask;
   
   %thr and connected components
   CC = bwconncomp(imDoG > thrDoG,6);
   
   xyz = zeros(CC.NumObjects,3);
   count = 0;
   for jj = 1:CC.NumObjects
      if( length(CC.PixelIdxList{jj}) < minSize || length(CC.PixelIdxList{jj}) > maxSize )
          continue;
      end
      
      count = count + 1;
      [x,y,z] = ind2sub(CC.ImageSize,CC.PixelIdxList{jj});
      xyz(count,:) = mean([x,y,z]);
   end
   
   xyz(count+1:end,:) = [];
   
   %Java indexing
   xyz = xyz -1;
   
   %write out interest points
   fid = fopen([imPath filesep 'interestpoints' filesep 'tpId_0_viewSetupId_' num2str(ii)  '.beadsM.ip.txt'],'w');
   fprintf(fid,'id\t x\t y\t z\n');
   fprintf(fid,'%d\t %f\t %f\t %f\n',[[0:size(xyz,1)-1]', xyz]');   
   fclose(fid);
end

%save parameters 
save([imPath filesep 'interestpoints' filesep 'tpId_0.beadsM.ip.mat'],'sigma','thrDoG','minSize','maxSize', 'thrMask');

