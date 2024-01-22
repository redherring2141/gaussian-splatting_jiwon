python ./train.py -s ../datasets/MipNeRF360/bicycle/ -m ./output/bicycle_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/bonsai/ -m ./output/bonsai_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/counter/ -m ./output/counter_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/flowers/ -m ./output/flowers_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/garden/ -m ./output/garden_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/kitchen/ -m ./output/kitchen_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/room/ -m ./output/room_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/stump/ -m ./output/stump_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/MipNeRF360/treehill/ -m ./output/treehill_7k --iterations 7_000 --eval


python ./train.py -s ../datasets/MipNeRF360/bicycle/ -m ./output/bicycle_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/bonsai/ -m ./output/bonsai_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000
python ./train.py -s ../datasets/MipNeRF360/counter/ -m ./output/counter_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/flowers/ -m ./output/flowers_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/garden/ -m ./output/garden_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/kitchen/ -m ./output/kitchen_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/room/ -m ./output/room_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/stump/ -m ./output/stump_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/MipNeRF360/treehill/ -m ./output/treehill_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
