%combine coarse and fine registration to generate the final stack (using a
%single interpolation)
%function script_simview3_final_registration()


baseRegistrationFolder = 'E:\simview3_deconvolution\15_04_24_fly_functionalImage\TM1445\Matlab_coarse_register_downsample2x_doubleBlurred'
%%
%load coarse registration
load([baseRegistrationFolder '\imRegister_Matlab_tform.mat'],'tformCell','imPath','imFilenameCell', 'numLevels');


