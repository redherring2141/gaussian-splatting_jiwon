for DATA in bicycle bonsai counter flowers garden kitchen room stump treehill
do
	python ./measurements/ncu_profile_20240401/analyze_ncucsv.py ./measurements/ncu_profile_20240401/ncu_render_$DATA\_30k.csv
done
