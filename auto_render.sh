#nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,memory.free,memory.used,utilization.gpu,utilization.memory --format=csv -lms 20 > ./smimeasure_20240327 &
PID=$!

for DATA in bicycle bonsai counter flowers garden kitchen room stump treehill
do
    python ./render_memprof.py -m ../output_full/$DATA\_30k --skip_train
done

#sleep 3s

#kill $PID
