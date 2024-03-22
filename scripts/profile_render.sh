pip install ./submodules/simple-knn/ ./submodules/diff-gaussian-rasterization/

python ./render_profiling.py -m ./output_test/bonsai_300_nonv_test > ./test/log_render_bonsai_300_nonv_test_20240112_0222

python ./measurements/analyze.py ./test/log_render_bonsai_300_nonv_test_20240112_0222 > ./test/summary_render_bonsai_300_nonv_test_20240112_0222
