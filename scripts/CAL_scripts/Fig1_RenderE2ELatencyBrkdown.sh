#!/bin/bash
function E2eLatencyBrkdown(){
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
	python3 ./render_profiling.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train --CUDAEVENT > /home/jetson-agx/NeRF/gaussian-splatting_jiwon/measurements/log_20240528_fig1/log_render_CUDAEVENT_$1_$2
	python3 ./Fig1_LogParser.py /home/jetson-agx/NeRF/gaussian-splatting_jiwon/measurements/log_20240528_fig1/log_render_CUDAEVENT_$1_$2
	python3 ./render.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train
	python3 ./metrics_profiling.py -m /home/jetson-agx/NeRF/models/$1/ > /home/jetson-agx/NeRF/gaussian-splatting_jiwon/measurements/log_20240528_fig1/log_metric_CUDAEVENT_$1_$2
	python3 ./Fig1_LogParser.py /home/jetson-agx/NeRF/gaussian-splatting_jiwon/measurements/log_20240528_fig1/log_metric_CUDAEVENT_$1_$2
fi
}

for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
#for DATA in room stump treehill mic chair ship materials lego drums ficus hotdog
do
	E2eLatencyBrkdown $DATA\_30k exp #$FUNC $PART
	#echo $DATA\_30k $FUNC $PART
done
