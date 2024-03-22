#!/bin/bash
if [ $# -lt 3 ];
then
	echo "$0: Missing arguments"
	exit 1
elif [ $# -gt 3 ];
then
	echo "$0: Too many arguments"
	exit 1
else
	echo "Processing $1_$2_$3 ..."
	cp ./submodules/diff-gaussian-rasterization/cuda_rasterizer/$2/forward_$3.cu ./submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
	pip install ./submodules/diff-gaussian-rasterization

	python ./render_profiling.py -m ../output_full/$1/ --skip_train --CUDAEVENT > ./measurements/log_20240312/log_render_CUDAEVENT_$1_$2_$3
	python ./measurements/analyze.py ./measurements/log_20240312/log_render_CUDAEVENT_$1_$2_$3
	python ./render.py -m ../output_full/$1/ --skip_train
	python ./metrics_profiling.py -m ../output_full/$1/ > ./measurements/log_20240312/log_metric_CUDAEVENT_$1_$2_$3
	python ./measurements/analyze.py ./measurements/log_20240312/log_metric_CUDAEVENT_$1_$2_$3
fi