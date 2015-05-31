%very simple procedure to get a rough mask
%you want to chose a threshold a little bit above backgroudn (~110 for wsimview 3)
function mask = maskEmbryo(im, thr)

radM = 15;%radius for morphological operations

mask = im >= thr;


%erose to disconnect beads form main embryo
se = strel('disk',radM,0);
for ii = 1:size(im,3)
   mask(:,:,ii) = imerode(mask(:,:,ii),se); 
end

%find largest connected component
CC = bwconncomp(mask,26);

ll = cellfun(@length, CC.PixelIdxList);


[~, pos] = max(ll);

mask = false(size(mask));
mask(CC.PixelIdxList{pos}) = true;


%dilate to extend embryo mask
se = strel('disk',radM + 10,0);
for ii = 1:size(im,3)
   mask(:,:,ii) = imdilate(mask(:,:,ii),se); 
end








