#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/output.txt"
log_file="/dev/null"

for i in `seq $1 $2`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	log_file="/groups/keller/home/amatf/temp/output_zebrafish_${i}.txt"
	echo ${i}
	cmd="/groups/keller/home/amatf/temp/multiview-deconvolution/CUDA/build_release/src/multiview_deconvolution_LR_multiGPU /nobackup/keller/deconvolution/Dre_HuC_20150709_170711.corrected/TM000${i}/MVref_deconv_LR_multiGPU_param_JFCluster_TM${i}.xml"
	
	qsub -pe batch 16 -l gpu_k20=true -b y -j y -o ${log_file} -V ${cmd}	
done    


