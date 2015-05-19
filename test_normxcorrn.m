%function test_normxcorrn()

%2D comparison
template = 1000 * rand(3,5);
im = 100 * rand(128);

fvA = normxcorr2(template, im);
fvB = normxcorrn(template, im);

figure;imagesc(fvA(2:end-1,3:end-2)-fvB);
colorbar;

%%
%3D run
template = rand(3,5,7);
im = 100 * rand(128,256,512);

fvB3 = normxcorrn(template, im);

%%
%test speed
template = rand(64,128);
im = rand(4096,4096);
tic;
fvA = normxcorr2(template, im);
toc;
tic;
fvB = normxcorrn(template, im);
toc;