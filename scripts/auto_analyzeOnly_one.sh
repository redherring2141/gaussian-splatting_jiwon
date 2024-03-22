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
	python ./measurements/analyze_blending.py ./measurements/log_20240312/log_render_CUDAEVENT_$1_$2_$3
	python ./measurements/analyze_blending.py ./measurements/log_20240312/log_metric_CUDAEVENT_$1_$2_$3
fi
