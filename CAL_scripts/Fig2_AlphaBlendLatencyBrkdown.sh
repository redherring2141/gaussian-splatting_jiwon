#!/bin/bash
function BlendingLatencyBrkdown(){
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
	python3 ./render_profiling.py -m ../models/$1/ --skip_train --CUDAEVENT > ./measurements/log_20240528_fig2/log_render_CUDAEVENT_$1_$2_$3
	python3 ./Fig2_LogParse.py ./measurements/log_20240528_fig2/log_render_CUDAEVENT_$1_$2_$3
	python3 ./render.py -m ../models/$1/ --skip_train
	python3 ./metrics_profiling.py -m ../models/$1/ > ./measurements/log_20240528_fig2/log_metric_CUDAEVENT_$1_$2_$3
	python3 ./Fig2_LogParse.py ./measurements/log_20240528_fig2/log_metric_CUDAEVENT_$1_$2_$3
fi
}

for PART in orgin fully noall noeq1 noeq2 noeq3 only1 only2 only3
do
    for FUNC in exp xpf epf
    do
    	cp ./submodules/diff-gaussian-rasterization/cuda_rasterizer/$FUNC/forward_$PART.cu ./submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
    	pip install ./submodules/diff-gaussian-rasterization

        for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
        do
            BlendingLatencyBrkdown $DATA\_30k $FUNC $PART
            #echo $DATA\_30k $FUNC $PART
        done
    done
done