param([Int]$start=0,[Int]$end=3)

$bindir="C:\Users\SiMView\cppProjects\multiview-deconvolution\CUDA\build\src\Release"
$cmd="main_multiviewDeconvLR_multiGPU_blocksZ.exe"

$datadir="S:\SiMView1\Temp\150810\SiMView-XMLs\WithPSFS"

function Timepoint($time) {
	$file = "TM{0}\MVref_deconv_LR_multiGPU_param_TM{1}.xml" -f ($time).toString("000000"),($time).toString()
	join-path $datadir $file
}

for($i=$start;$i -le $end;$i++) {
	&(join-path $bindir $cmd) (Timepoint $i)
}



