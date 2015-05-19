%function test_normxcorrn()

template = rand(3,5);
im = rand(128);

fvA = normxcorr2(template, im);
fvB = normxcorrn(template, im);

figure;imagesc(fvA(2:end-1,3:end-2)-fvB);
colorbar;


%test speed
template = rand(64,128);
im = rand(4096,4096);
tic;
fvA = normxcorr2(template, im);
toc;
tic;
fvB = normxcorrn(template, im);
toc;