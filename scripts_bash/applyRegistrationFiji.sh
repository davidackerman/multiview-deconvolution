#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/outputFiji"
log_file="/dev/null"

for i in `seq 643 643`;
do
	j=${i}
	if [ $i -le 999 ];
    then
		j="0${i}"
	fi
    

	echo ${i}

	#membrane channel
	cmd="/groups/keller/home/amatf/multiview-deconvolution/scripts_bash/xvfb-run -a /groups/keller/home/amatf/Fiji.app/ImageJ-linux64 -Xms100g -Xmx100g -- --no-splash -macro /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}membrCh_Fiji/Macro.ijm"	
	qsub -pe batch 16 -l short=true -b y -cwd -j y -o ${log_file} -V ${cmd}	

	
	#nuclear channel
         cmd="/groups/keller/home/amatf/multiview-deconvolution/scripts_bash/xvfb-run -a /groups/keller/home/amatf/Fiji.app/ImageJ-linux64 -Xms100g -Xmx100g -- --no-splash -macro /nobackup/keller/15_06_11_fly_functionalImage/TM00${j}_Fiji/Macro.ijm"
        qsub -pe batch 16 -l short=true -b y -cwd -j y -o ${log_file} -V ${cmd}

done    


