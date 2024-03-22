#!/bin/bash


for PART in orgin #fully noall noeq1 noeq2 noeq3 only1 only2 only3
do
    for FUNC in exp epf xpf
    do
    	cp ./submodules/diff-gaussian-rasterization/cuda_rasterizer/$FUNC/forward_$PART.cu ./submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
    	pip install ./submodules/diff-gaussian-rasterization

        for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
        do
            sh ./auto_blending_profile_one.sh $DATA\_30k $FUNC $PART
            #echo $DATA\_30k $FUNC $PART
        done
    done
done

