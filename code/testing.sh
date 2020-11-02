x=1
while [ $x -le 1 ]
do
  echo "iteration $x"
  x=$(( $x + 1 ))
    for layer in 1 2 3 
        # for length in 100 
    do
        for sparcity in 0 100
        # for sparcity in 10
        do
            sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --csv \
            --log-file /home/homdev-nano/addProfiling/PyTorch-GPU-Profiling/monteCarlo/mnist/FLOPS_mnist_layer${layer}_sparcityPercent_${sparcity}_iteration${x}.csv \
            --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_hp_mul,\
            flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency python3 mnist.py $layer $sparcity $x 
                    
            sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof \
            --log-file /home/homdev-nano/addProfiling/PyTorch-GPU-Profiling/monteCarlo/mnist/InputDepthTrace_mnist_layer${layer}_sparcityPercent_${sparcity}_iteration${x}.csv --print-gpu-trace --csv \
            python3 mnist.py $layer $sparcity $x 
        done
    done
    
    for layer in 1 2 3 4 5
    do
        for sparcity in 0 100
        # for sparcity in 10
        do
            sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --csv \
            --log-file /home/homdev-nano/addProfiling/PyTorch-GPU-Profiling/monteCarlo/celeba/FLOPS_mnist_layer${layer}_sparcityPercent_${sparcity}_iteration${x}.csv \
            --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_hp_mul,\
            flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency python3 celeba.py $layer $sparcity $x 
                    
            sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof \
            --log-file /home/homdev-nano/addProfiling/PyTorch-GPU-Profiling/monteCarlo/celeba/InputDepthTrace_mnist_layer${layer}_sparcityPercent_${sparcity}_iteration${x}.csv --print-gpu-trace --csv \
            python3 celeba.py $layer $sparcity $x 
        done
    done

    for layer in 1 2 3 4 5 6
    do
        for sparcity in 0 100
        do
            sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof --csv \
            --log-file /home/homdev-nano/addProfiling/PyTorch-GPU-Profiling/monteCarlo/lsun/FLOPS_mnist_layer${layer}_sparcityPercent_${sparcity}_iteration${x}.csv \
            --metrics achieved_occupancy,ipc,l2_read_throughput,l2_write_throughput,sm_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_hp_mul,\
            flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_dp_efficiency,flop_hp_efficiency,flop_sp_efficiency python3 lsun.py $layer $sparcity $x 
                    
            sudo -HE env PATH=$PATH PYTHONPATH=$PYTHONPATH /usr/local/cuda-10.0/bin/nvprof \
            --log-file /home/homdev-nano/addProfiling/PyTorch-GPU-Profiling/monteCarlo/lsun/InputDepthTrace_mnist_layer${layer}_sparcityPercent_${sparcity}_iteration${x}.csv --print-gpu-trace --csv \
            python3 lsun.py $layer $sparcity $x 
        done
    done

done


