for DATA in bicycle bonsai counter flowers garden kitchen room stump treehill
do
    ncu -i ./ncu_render_$DATA\_30k.ncu-rep --csv --page details --metrics sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_targets_threads_uniform.pct > ncu_render_$DATA\_30k.csv
done