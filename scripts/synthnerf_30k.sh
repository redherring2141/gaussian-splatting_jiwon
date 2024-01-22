python ./train.py -s ../datasets/nerf_synthetic/lego -m ./output/lego --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/lego
python ./metrics.py -m ./output/lego

python ./train.py -s ../datasets/nerf_synthetic/chair -m ./output/chair --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/chair
python ./metrics.py -m ./output/chair

python ./train.py -s ../datasets/nerf_synthetic/drums -m ./output/drums --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/drums
python ./metrics.py -m ./output/drums

python ./train.py -s ../datasets/nerf_synthetic/ficus -m ./output/ficus --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/ficus
python ./metrics.py -m ./output/ficus

python ./train.py -s ../datasets/nerf_synthetic/hotdog -m ./output/hotdog --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/hotdog
python ./metrics.py -m ./output/hotdog

python ./train.py -s ../datasets/nerf_synthetic/materials -m ./output/materials --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/materials
python ./metrics.py -m ./output/materials

python ./train.py -s ../datasets/nerf_synthetic/mic -m ./output/mic --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/mic
python ./metrics.py -m ./output/mic

python ./train.py -s ../datasets/nerf_synthetic/ship -m ./output/ship --test_iterations 30000 --save_iterations 30000 --eval
python ./render.py -m ./output/ship
python ./metrics.py -m ./output/ship
