for DATA in fern flower fortress horns leaves orchids room trex
do
    python ./train.py -s ../datasets/nerf_llff_data/undistorted/$DATA -m ../output_full/$DATA\_30k --iter 30000
    python ./render.py -m ../output_full/$DATA\_30k
    python ./render_FPS.py -m ../output_full/$DATA\_30k
    python ./metrics.py -m ../output_full/$DATA\_30k
done