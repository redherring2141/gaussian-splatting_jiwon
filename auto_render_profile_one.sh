#!/bin/bash
if [ $# -lt 2 ];
then
	echo "$0: Missing arguments"
	exit 1
elif [ $# -gt 2 ];
then
	echo "$0: Too many arguments"
	exit 1
else
	echo "Processing $1 ..."
	python ./render_profiling.py -m /home/jiwon/NeRF/output_full/$1/ --skip_train --CUDAEVENT > /home/jiwon/NeRF/gaussian-splatting_jiwon/measurements/log_20240303/log_render_CUDAEVENT_$1_$2
	python ./measurements/analyze.py /home/jiwon/NeRF/gaussian-splatting_jiwon/measurements/log_20240303/log_render_CUDAEVENT_$1_$2
	python ./render.py -m /home/jiwon/NeRF/output_full/$1/ --skip_train
	python ./metrics_profiling.py -m /home/jiwon/NeRF/output_full/$1/ > /home/jiwon/NeRF/gaussian-splatting_jiwon/measurements/log_20240303/log_metric_CUDAEVENT_$1_$2
	python ./measurements/analyze.py /home/jiwon/NeRF/gaussian-splatting_jiwon/measurements/log_20240303/log_metric_CUDAEVENT_$1_$2
fi
