python ./render_profiling.py -m ./output_full/train_30k/ --skip_train > ./measurements/log/log_tandt_train_30k
python ./measurements/analyze.py ./measurements/log/log_tandt_train_30k > ./measurements/summary/summary_tandt_train_30k.txt

python ./render_profiling.py -m ./output_full/truck_30k/ --skip_train > ./measurements/log/log_tandt_truck_30k
python ./measurements/analyze.py ./measurements/log/log_tandt_truck_30k > ./measurements/summary/summary_tandt_truck_30k.txt


python ./render_profiling.py -m ./output_full/drjohnson_30k/ --skip_train > ./measurements/log/log_db_drjohnson_30k
python ./measurements/analyze.py ./measurements/log/log_db_drjohnson_30k > ./measurements/summary/summary_db_drjohnson_30k.txt

python ./render_profiling.py -m ./output_full/playroom_30k/ --skip_train > ./measurements/log/log_db_playroom_30k
python ./measurements/analyze.py ./measurements/log/log_db_playroom_30k > ./measurements/summary/summary_db_playroom_30k.txt



python ./render_profiling.py -m ./output_full/mic_30k/ --skip_train > ./measurements/log/log_synth_mic_30k
python ./measurements/analyze.py ./measurements/log/log_synth_mic_30k > ./measurements/summary/summary_synth_mic_30k.txt

python ./render_profiling.py -m ./output_full/chair_30k/ --skip_train > ./measurements/log/log_synth_chair_30k
python ./measurements/analyze.py ./measurements/log/log_synth_chair_30k > ./measurements/summary/summary_synth_chair_30k.txt

python ./render_profiling.py -m ./output_full/ship_30k/ --skip_train > ./measurements/log/log_synth_ship_30k
python ./measurements/analyze.py ./measurements/log/log_synth_ship_30k > ./measurements/summary/summary_synth_ship_30k.txt

python ./render_profiling.py -m ./output_full/materials_30k/ --skip_train > ./measurements/log/log_synth_materials_30k
python ./measurements/analyze.py ./measurements/log/log_synth_materials_30k > ./measurements/summary/summary_synth_materials_30k.txt

python ./render_profiling.py -m ./output_full/lego_30k/ --skip_train > ./measurements/log/log_synth_lego_30k
python ./measurements/analyze.py ./measurements/log/log_synth_lego_30k > ./measurements/summary/summary_synth_lego_30k.txt

python ./render_profiling.py -m ./output_full/drums_30k/ --skip_train > ./measurements/log/log_synth_drums_30k
python ./measurements/analyze.py ./measurements/log/log_synth_drums_30k > ./measurements/summary/summary_synth_drums_30k.txt

python ./render_profiling.py -m ./output_full/ficus_30k/ --skip_train > ./measurements/log/log_synth_ficus_30k
python ./measurements/analyze.py ./measurements/log/log_synth_ficus_30k > ./measurements/summary/summary_synth_ficus_30k.txt

python ./render_profiling.py -m ./output_full/hotdog_30k/ --skip_train > ./measurements/log/log_synth_hotdog_30k
python ./measurements/analyze.py ./measurements/log/log_synth_hotdog_30k > ./measurements/summary/summary_synth_hotdog_30k.txt




python ./render_profiling.py -m ./output_full/bicycle_30k/ --skip_train > ./measurements/log/log_mip360_bicycle_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_bicycle_30k > ./measurements/summary/summary_mip360_bicycle_30k.txt

python ./render_profiling.py -m ./output_full/bonsai_30k/ --skip_train > ./measurements/log/log_mip360_bonsai_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_bonsai_30k > ./measurements/summary/summary_mip360_bonsai_30k.txt

python ./render_profiling.py -m ./output_full/counter_30k/ --skip_train > ./measurements/log/log_mip360_counter_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_counter_30k > ./measurements/summary/summary_mip360_counter_30k.txt

python ./render_profiling.py -m ./output_full/flowers_30k/ --skip_train > ./measurements/log/log_mip360_flowers_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_flowers_30k > ./measurements/summary/summary_mip360_flowers_30k.txt

python ./render_profiling.py -m ./output_full/garden_30k/ --skip_train > ./measurements/log/log_mip360_garden_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_garden_30k > ./measurements/summary/summary_mip360_garden_30k.txt

python ./render_profiling.py -m ./output_full/kitchen_30k/ --skip_train > ./measurements/log/log_mip360_kitchen_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_kitchen_30k > ./measurements/summary/summary_mip360_kitchen_30k.txt

python ./render_profiling.py -m ./output_full/room_30k/ --skip_train > ./measurements/log/log_mip360_room_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_room_30k > ./measurements/summary/summary_mip360_room_30k.txt

python ./render_profiling.py -m ./output_full/stump_30k/ --skip_train > ./measurements/log/log_mip360_stump_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_stump_30k > ./measurements/summary/summary_mip360_stump_30k.txt

python ./render_profiling.py -m ./output_full/treehill_30k/ --skip_train > ./measurements/log/log_mip360_treehill_30k
python ./measurements/analyze.py ./measurements/log/log_mip360_treehill_30k > ./measurements/summary/summary_mip360_treehill_30k.txt
