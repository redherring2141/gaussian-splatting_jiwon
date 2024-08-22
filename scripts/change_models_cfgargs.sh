# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
# do
# 	cd /home/jetson-agx/NeRF/models/$DATA\_30k/
# 	sed -i 's/work6\/jiwon/home\/jetson-agx/g' cfg_args
# 	sed -i 's/.\/output/\/home\/jetson-agx\/NeRF\/models/g' cfg_args
# done

# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cd /work5/jiwon/NeRF/models_pretrained/$DATA\_30k
# 	sed -i 's/\/eval//g' cfg_args
# 	sed -i 's/f:\/bkerbl\/Downloads\/360_v2/\/work5\/jiwon\/NeRF\/datasets/g' cfg_args
# done

# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cd /work5/jiwon/NeRF/models_pretrained/$DATA\_30k
# 	sed -i "s/model_path='./model_path='\/work5\/jiwon\/NeRF\/models_pretrained/g" cfg_args
# done

# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cd /work5/jiwon/NeRF/models_pretrained/$DATA\_30k
# 	#sed -i "s/', resolution/_30k', resolution/g" cfg_args
# 	sed -i 's/f:\/bkerbl\/Downloads\/db/\/work5\/jiwon\/NeRF\/datasets/g' cfg_args
# 	sed -i 's/f:\/bkerbl\/Downloads\/tandt/\/work5\/jiwon\/NeRF\/datasets/g' cfg_args
# done



# cd /work5/jiwon/NeRF/models_pretrained/
# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cp -r $DATA\_7k
# 	mv $DATA $DATA\_30k
# 	rm -rf $DATA\_7k/point_cloud/iteration_30000
# 	rm -rf $DATA\_30k/point_cloud/iteration_7000
# done


# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cd /work5/jiwon/NeRF/models_pretrained/$DATA\_30k
# 	sed -i "s/Namespace(/Namespace(data_device='cuda', /g" cfg_args
# done

# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cp /work6/jiwon/NeRF/datasets/MipNeRF360/$DATA/transforms* /work5/jiwon/NeRF/datasets/$DATA
# 	cp /work6/jiwon/NeRF/datasets/DeepBlending/$DATA/transforms* /work5/jiwon/NeRF/datasets/$DATA
# 	cp /work6/jiwon/NeRF/datasets/Tanks_and_Temples/$DATA/transforms* /work5/jiwon/NeRF/datasets/$DATA
# done

# for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill
# do
# 	cp /work6/jiwon/NeRF/output_full/$DATA\_30k/$DATA\_30k.ingp /work5/jiwon/NeRF/models_pretrained/$DATA\_30k/
# done

for DATA in playroom drjohnson train truck counter bonsai room kitchen treehill flowers stump garden bicycle 
do
	cd /home/jetson-agx/NeRF/models_pretrained/$DATA\_30k/
	sed -i 's/work5\/jiwon/home\/jetson-agx/g' cfg_args
done
