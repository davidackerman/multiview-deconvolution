#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/output.txt"
log_file="/dev/null"

for i in `seq 450 1100`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	echo ${i}
	if [ ! -f /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}_Fiji/test_mv_deconv_LR_multiGPU_iter40.raw ]; then
	cmd="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/test_multiview_deconvolution_LR_multiGPU /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}_Fiji/ 40"
	
	qsub -pe batch 16 -l gpu_k20=true -b y -j y -o ${log_file} -V ${cmd}	
	fi

	 if [ ! -f /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}membrCh_Fiji/test_mv_deconv_LR_multiGPU_iter40.raw ]; then
	cmd="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/test_multiview_deconvolution_LR_multiGPU /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}membrCh_Fiji/ 40"
	 qsub -pe batch 16 -l gpu_k20=true -b y -j y -o ${log_file} -V ${cmd}
	fi
done    


