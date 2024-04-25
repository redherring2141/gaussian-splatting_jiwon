for DATA in fern flower fortress horns leaves orchids room trex
do
    colmap image_undistorter \
        --image_path ../$DATA/images \
        --input_path ../$DATA/sparse/0 \
        --output_path ./$DATA \
        --output_type COLMAP
    mkdir ./$DATA/sparse/0
    mv ./$DATA/sparse/*.bin ./$DATA/sparse/0/
done