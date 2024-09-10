#!/bin/bash

str1="A100"
str2="Xavier"

function E2eMemBrkdown(){
if [ $# -lt 3 ];
then
	echo "$0: Missing arguments"
	exit 1
elif [ $# -gt 3 ];
then
	echo "$0: Too many arguments"
	exit 1
else
	echo "Processing $1 on $3..."

	if [ $3 == $str1 ];
	then	
		#A100
		python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render_memprof.py -m /work6/jiwon/NeRF/output_full/$1/ --skip_train
		mkdir /work6/jiwon/NeRF/CAL_3DGS_rev/fig4_A100_20240818/$1
		mv ./*.html ./*.pickle ./*.gz /work6/jiwon/NeRF/CAL_3DGS_rev/fig4_A100_20240818/$1/
	else
		#Xavier-AGX
		python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render_profiling.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train
		mkdir /home/jetson-agx/NeRF/CAL_3DGS_rev/fig4_A100_20240818/$1
		mv ./*.html ./*.pickle ./*.gz /home/jetson-agx/NeRF/CAL_3DGS_rev/fig4_A100_20240818/$1/
	fi
fi
}

if [ $1 == $str1 ];
then
	#A100
	cp /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_org.cu /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
	pip install /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization
else
	#XavierAGX
	cp /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_org.cu /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
	pip install /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization
fi

for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
do
	E2eMemBrkdown $DATA\_30k exp $1 #$FUNC $PART
done
#String trim parameter input is 52 for A100 52 for Xavier