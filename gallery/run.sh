#!/bin/bash



# ===================================

# TYPE="single"
TYPE="multi"
network="tiny_conv_1"
batch_size=1
layout="NHWC"

# ===================================



case "${TYPE}" in
  single) py_file="cuda_singletask.py" ;;
  multi)  py_file="tune_network_cuda.py" ;;
  *) echo "Invalid type. Use 'single' or 'multi'." ; exit 1 ;;
esac

echo "======================================================"
echo "Running with follows :"
echo "   Type: ${TYPE}"
echo "   Network: ${network}"
echo "   Batch_size: ${batch_size}"
echo "   Layout: ${layout}"
echo "======================================================"


NSYS="${1:-}"

ts="$(date '+%m%d_%H%M')"
logs_nsys="logs_nsys/${TYPE}_${network}-b${batch_size}-${ts}"
logs_overall="logs_overall/${TYPE}_${network}/${TYPE}_${network}-b${batch_size}-${ts}.log"

mkdir -p "logs_overall/${TYPE}_${network}"


if [ "$NSYS" == "nsys" ]; then
    nsys profile --trace=cuda,nvtx,osrt \
        -o ${logs_nsys} \
        python ${py_file} \
        --network ${network} \
        --batch_size ${batch_size} \
        --layout ${layout} 2>&1 | tee ${logs_overall}
else
    rlwrap python ${py_file} \
        --network ${network} \
        --batch_size ${batch_size} \
        --layout ${layout} 2>&1 | tee ${logs_overall}
fi