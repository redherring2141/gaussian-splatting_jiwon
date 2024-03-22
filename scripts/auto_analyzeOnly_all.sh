#!/bin/bash


for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
do
    for FUNC in exp epf xpf
    do
        for PART in fully noall noeq1 noeq2 noeq3 only1 only2 only3 orgin
        do
            sh ./auto_analyzeOnly_one.sh $DATA\_30k $FUNC $PART
            #echo $DATA\_30k $FUNC $PART
        done
    done
done

