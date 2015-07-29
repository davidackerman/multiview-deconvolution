#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/output.txt"
log_file="/dev/null"

for i in `seq 3009 3057`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	echo ${i}
	cmd="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/test_multiview_deconvolution_LR_multiGPU /nobackup/keller/TM00${j}/simview3_TM${i}_ 40"
	
	qsub -pe batch 16 -l gpu_k20=true -b y -j y -o ${log_file} -V ${cmd}	
done    


