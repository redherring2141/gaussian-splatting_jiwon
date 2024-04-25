for DATA in bicycle bonsai counter flowers garden kitchen room stump treehill
do
	python ./ncu_profile_20240318/analyze_blending.py ./ncu_profile_20240318/ncu_render_$DATA\_30k.csv
done
