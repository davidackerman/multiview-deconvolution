#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/output.txt"
log_file="/dev/null"

for i in `seq 201 300`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	#uncomment this line if you want to save a log file
	log_file="/groups/keller/home/amatf/temp/output_mouse_15_04_03_TM${i}.txt"
	echo ${i}
	cmd="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/src/main_multiviewDeconvLR_multiGPU_blocksZ /nobackup/keller/deconvolution/Mmu_E1_mKate2_0_20150403_151711.corrected_ReallyUnmasked/SPM00/TM000${i}/MVref_deconv_LR_multiGPU_param_JFCluster_TM${i}.xml"
	
	qsub -pe batch 16 -l gpu_k20=true -b y -j y -o ${log_file} -V ${cmd}	
done    


