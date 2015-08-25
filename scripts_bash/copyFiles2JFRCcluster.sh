#!/bin/bash

for i in `seq 2620 2670`;
do
      cmd="cp -r /cygdrive/t/temp/deconvolution/15_05_12_fly_functionalImage_cluster/TM00${i} /cygdrive/a/deconvolutionDatasets/15_05_12_fly_functionalImage_cluster"
	  ${cmd}
done    