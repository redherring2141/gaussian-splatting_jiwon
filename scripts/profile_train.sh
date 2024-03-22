pip install ./submodules/simple-knn/ ./submodules/diff-gaussian-rasterization/

python ./train_profiling.py -s ../datasets/MipNeRF360/bonsai/ -m ./output_test/bonsai_300_nonv_test --iteration 300 --eval > ./test/log_train_bonsai_300_nonv_test_20240112_0222

python ./measurements/analyze.py ./test/log_train_bonsai_300_nonv_test_20240112_0222 > ./test/summary_train_bonsai_300_nonv_test_20240112_0222
