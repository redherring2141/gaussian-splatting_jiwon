for DATA in bicycle bonsai counter flowers garden kitchen room stump treehill
do
    ncu -i ./ncu_render_$DATA\_30k.ncu-rep --csv --page details --metrics thread_inst_executed,thread_inst_executed_true,inst_executed > ncu_render_$DATA\_30k.csv
done
