#!/bin/bash
function get_cuda_index(){
    cuda_uuid_array=(`echo $CUDA_VISIBLE_DEVICES | tr ','  '\n'`)
    ret=""
    for cuda_uuid in ${cuda_uuid_array[*]}
    do
        if [[ ${cuda_uuid} == GPU* ]]
        then
            cuda_index=`nvidia-smi -L |grep ${cuda_uuid} |grep -o '[0-9]\d*:'|grep -o '[0-9]\d*'`
        else
            cuda_index=${cuda_uuid}
        fi
        
        
        if [ ${#ret} -gt 0 ]
        then
            ret="${ret},${cuda_index}"
        else
            ret="${cuda_index}"
        fi
    done
}
echo "original CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
get_cuda_index
echo "CUDA_VISIBLE_DEVICES=$ret"

export CUDA_VISIBLE_DEVICES=$ret && python $(dirname $0)/mnist_lenet.py