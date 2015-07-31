#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/outputFijiTiff2KLB"
log_file="/dev/null"

for i in `seq 450 1100`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	echo ${i}

	#nuclear channel
	if [ ! -f /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}_Fiji/weightsReg_4.klb ]; then
		cmd="/groups/keller/home/amatf/multiview-deconvolution/tiff2klb/build_release/tiff2klb /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}_Fiji/"	
		qsub -pe batch 16 -l short=true -b y -cwd -j y -o ${log_file} -V ${cmd}	
	fi
	


	#membrane channel
	if [ ! -f /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}membrCh_Fiji/weightsReg_4.klb ]; then
		cmd="/groups/keller/home/amatf/multiview-deconvolution/tiff2klb/build_release/tiff2klb /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}membrCh_Fiji/"
	        qsub -pe batch 16 -l short=true -b y -cwd -j y -o ${log_file} -V ${cmd}
	fi

done    


