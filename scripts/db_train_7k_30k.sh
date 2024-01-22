python ./train.py -s ../datasets/DeepBlending/drjohnson -m ./output/drjohnson_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/DeepBlending/playroom/ -m ./output/playroom_7k --iterations 7_000 --eval


python ./train.py -s ../datasets/DeepBlending/drjohnson -m ./output/drjohnson_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/DeepBlending/playroom/ -m ./output/playroom_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000
