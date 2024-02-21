#Nsight-Compute-CLI command with a specific measurement

#sudo /usr/local/cuda/bin/nv-nsight-cu-cli --metrics smsp__sass_average_branch_targets_threads_uniform.pct -o ncu_render_bonsai30k_20240213_2300 /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python ./render_profiling.py -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train --NSYSNVTX

#Nsight-Compute-CLI command with overall measurements
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o ncu_render_bonsai30k_20240214_1231 --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train --NSYSNVTX


#Nsight-Compute-CLI command with range setting
#sudo /usr/local/cuda-11.6/bin/nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o bonsai_30k_render_20240206 -w true /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o bonsai_30k_render_nsightcompute --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu -o bonsai_30k_render_nsightcompute --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o bonsai_30k_render_nvcu_20240206 --capture-range=cudaProfilerApi --stop-on-rang-end=true --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o bonsai_30k_render_nvcu_20240206 --profile-from-start off --capture-range=cudaProfilerApi --stop-on-range-end=true --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o bonsai_30k_render_nvcu_20240206 --profile-from-start off --capture-range=cudaProfilerAPI --stop-on-range-end=true --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
#sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o bonsai_30k_render_nvcu_20240206 --profile-from-start off --range=1-20 --stop-on-range-end=true --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train
sudo /usr/local/cuda-11.6/bin/nv-nsight-cu-cli -o ncu_render_bonsai_30k_20240214_1305 --profile-from-start off --range=1-20 --set full /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py --NSYSNVTX -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train




#Nsight-System command for rendering measurement
#sudo /usr/local/cuda-11.6/bin/nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o 3dgs_render_test -w true /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/render_profiling.py -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_full/bonsai_30k/ --skip_train

#Nsight-System command for training measurement with logging
#sudo /usr/local/cuda-11.6/bin/nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o bonsai_30k_train_test -w true /home/jiwon/miniconda3/envs/gaussian_splatting/bin/python /home/jiwon/NeRF/gaussian-splatting_jiwon/train_profiling.py -s /home/jiwon/NeRF/datasets/MipNeRF360/bonsai/ -m /home/jiwon/NeRF/gaussian-splatting_jiwon/output_test/bonsai_nvtx_test_30k --eval > ~/log_train_bonsai_30k_test_20240111_0342

#Message logging with tee pipe
#python ./train_profiling.py -s ../datasets/nerf_synthetic/materials/ -m ./output_test/material_test_30k --eval | tee -i ./log_train_test_20240109_2242
