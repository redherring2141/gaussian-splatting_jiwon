for DATA in drjohnson playroom train truck bicycle bonsai counter flowers garden kitchen room stump treehill mic chair ship materials lego drums ficus hotdog
do
	cd /home/jetson-agx/NeRF/models/$DATA\_30k/
	sed -i 's/work6\/jiwon/home\/jetson-agx/g' cfg_args
	sed -i 's/.\/output/\/home\/jetson-agx\/NeRF\/models/g' cfg_args
done
