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
	#A100
	python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render_profiling.py -m /work6/jiwon/NeRF/models/$1/ --skip_train --CUDAEVENT > ./fig3_A100_20240813/log_render_CUDAEVENT_$1_$2_$3
	python3 ./Fig3_LogParse.py ./fig3_A100_20240813/log_render_CUDAEVENT_$1_$2_$3 ./fig3_A100_20240813/summary_blending.txt
	python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render.py -m /work6/jiwon/NeRF/models/$1/ --skip_train
	python3 /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/metrics_profiling.py -m /work6/jiwon/NeRF/models/$1/ > ./fig3_A100_20240813/log_metric_CUDAEVENT_$1_$2_$3
	python3 ./Fig3_LogParse.py ./fig3_A100_20240813/log_metric_CUDAEVENT_$1_$2_$3 ./fig3_A100_20240813/summary_blending.txt

	#Xavier-AGX
	# python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render_profiling.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train --CUDAEVENT > ./fig3_XavierAGX_20240813/log_render_CUDAEVENT_$1_$2_$3
	# python3 ./Fig3_LogParse.py ./fig3_XavierAGX_20240813/log_render_CUDAEVENT_$1_$2_$3 ./fig3_XavierAGX_20240813/summary_blending.txt
	# python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/render.py -m /home/jetson-agx/NeRF/models/$1/ --skip_train
	# python3 /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/metrics_profiling.py -m /home/jetson-agx/NeRF/models/$1/ > ./fig3_XavierAGX_20240813/log_metric_CUDAEVENT_$1_$2_$3
	# python3 ./Fig3_LogParse.py ./fig3_XavierAGX_20240813/log_metric_CUDAEVENT_$1_$2_$3 ./fig3_XavierAGX_20240813/summary_blending.txt
fi
}

# for PART in orgin fully noall noeq1 noeq2 noeq3 only1 only2 only3
# do
#     for FUNC in exp xpf epf
#     do
#     	cp ./submodules/diff-gaussian-rasterization/cuda_rasterizer/$FUNC/forward_$PART.cu ./submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
#     	pip install ./submodules/diff-gaussian-rasterization

#         for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
#         do
#             BlendingLatencyBrkdown $DATA\_30k $FUNC $PART
#             #echo $DATA\_30k $FUNC $PART
#         done
#     done
# done

for PART in orgin fully noall noeq1 noeq2 noeq3 only1 only2 only3
do
    for FUNC in exp xpf epf
    do
		#A100
    	cp /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/$FUNC/forward_$PART.cu /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
    	pip install /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization

		#XavierAGX
    	# cp /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/$FUNC/forward_$PART.cu /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
    	# pip install /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization

        for DATA in bicycle garden kitchen room stump treehill
        do
            BlendingLatencyBrkdown $DATA\_30k $FUNC $PART
            #echo $DATA\_30k $FUNC $PART
        done
    done
done

#A100
cp /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_org.cu /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
pip install /work6/jiwon/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization

#XavierAGX
# cp /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_org.cu /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
# pip install /home/jetson-agx/NeRF/CAL_3DGS_rev/gaussian-splatting_jiwon/submodules/diff-gaussian-rasterization