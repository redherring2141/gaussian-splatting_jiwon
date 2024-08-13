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
	#A100
	python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render_profiling.py -m /work6/jiwon/NeRF/output_full/$1/ --skip_train --CUDAEVENT > ./fig2_A100_20240813/log_render_CUDAEVENT_$1_$2
	python3 ./Fig2_LogParser.py ./fig2_A100_20240813/log_render_CUDAEVENT_$1_$2 ./fig2_A100_20240813/summary_render_CUDAEVENT.txt
	python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render.py -m /work6/jiwon/NeRF/output_full/$1/ --skip_train
	python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/metrics_profiling.py -m /work6/jiwon/NeRF/output_full/$1/ > ./fig2_A100_20240813/log_metric_CUDAEVENT_$1_$2
	python3 ./Fig2_LogParser.py ./fig2_A100_20240813/log_metric_CUDAEVENT_$1_$2 ./fig2_A100_20240813/summary_render_CUDAEVENT.txt

	#Xavier-AGX
	# python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render_profiling.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train --CUDAEVENT > ./fig2_XavierAGX_20240812/log_render_CUDAEVENT_$1_$2
	# python3 ./Fig2_LogParser.py ./fig2_XavierAGX_20240812/log_render_CUDAEVENT_$1_$2 ./fig2_XavierAGX_20240812/summary_render_CUDAEVENT.txt
	# python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train
	# python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/metrics_profiling.py -m /home/jetson-agx/NeRF/models/$1/ > ./fig2_XavierAGX_20240812/log_metric_CUDAEVENT_$1_$2
	# python3 ./Fig2_LogParser.py ./fig2_XavierAGX_20240812/log_metric_CUDAEVENT_$1_$2 ./fig2_XavierAGX_20240812/summary_render_CUDAEVENT.txt
fi
}

#for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
for DATA in train truck
do
	E2eLatencyBrkdown $DATA\_30k exp #$FUNC $PART
	#echo $DATA\_30k $FUNC $PART
done