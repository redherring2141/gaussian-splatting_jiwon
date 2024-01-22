python ./train.py -s ../datasets/nerf_synthetic/mic/ -m ./output/mic_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/chair/ -m ./output/chair_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/ship/ -m ./output/ship_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/materials/ -m ./output/materials_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/lego/ -m ./output/lego_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/drums/ -m ./output/drums_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/ficus/ -m ./output/ficus_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/nerf_synthetic/hotdog/ -m ./output/hotdog_7k --iterations 7_000 --eval


python ./train.py -s ../datasets/nerf_synthetic/mic/ -m ./output/mic_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/nerf_synthetic/chair/ -m ./output/chair_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000
python ./train.py -s ../datasets/nerf_synthetic/ship/ -m ./output/ship_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/nerf_synthetic/materials/ -m ./output/materials_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/nerf_synthetic/lego/ -m ./output/lego_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/nerf_synthetic/drums/ -m ./output/drums_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/nerf_synthetic/ficus/ -m ./output/ficus_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/nerf_synthetic/hotdog/ -m ./output/hotdog_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
