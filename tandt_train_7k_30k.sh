python ./train.py -s ../datasets/Tanks_and_Temples/train/ -m ./output/train_7k --iterations 7_000 --eval
python ./train.py -s ../datasets/Tanks_and_Temples/truck/ -m ./output/truck_7k --iterations 7_000 --eval


python ./train.py -s ../datasets/Tanks_and_Temples/train/ -m ./output/train_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000 
python ./train.py -s ../datasets/Tanks_and_Temples/truck/ -m ./output/truck_30k --iterations 30_000 --eval --test_iterations 30000 --save_iterations 30000
