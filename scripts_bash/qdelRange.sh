#!/bin/bash


#log_file="/groups/keller/home/amatf/multiview-deconvolution/CUDA/build_release/test/output.txt"
log_file="/dev/null"

for i in `seq $1 $2`;
do
    

	qdel ${i}
done    


