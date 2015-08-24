#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/output.txt"
log_file="/dev/null"

for i in `seq 10001 10016`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	log_file="/groups/keller/home/amatf/temp/output_fly_beads_${i}.txt"
	echo ${i}
	cmd="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/src/multiview_deconvolution_LR_multiGPU /nobackup/keller/deconvolution/20150522_160119_fly_with_beads_TM000002/XMLfiles_multiGPUdeconv_cluster/beads_${i}.xml"
	
	qsub -pe batch 16 -l gpu_k20=true -b y -j y -o ${log_file} -V ${cmd}	
done    


