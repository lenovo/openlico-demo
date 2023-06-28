This project contains scripts for submitting jobs

- Under the hpc folder are for HPC jobs, including MPI (also available for Intel-MPI) and OpenMP (also available for Intel-OpenMP)
- Under the ai folder are for AI jobs, including caffe, intel-caffe, tensorflow, mxnet, neon, pytorch, paddlepaddle and other frameworks
- Under the intel_oneapi folder are for the Intel-OneAPI jobs, including Intel-GBD, Intel-Modin, etc.

Before using the scripts in this project, you need to prepare the dataset (the directory name is datasets, placed in the same directory as ai, hpc and intel_oneapi). datasets refers to the following directory structure to place datasets:

```
datasets
├── cifar10
│   ├── cifar-10-batches-bin  # download and extract from http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
│   └── cifar-10-batches-py   # download and extract from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
└── mnist
    ├── mnist.npz                       # download from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    ├── mnist_lmdb                      # follow the tutorial of https://github.com/BVLC/caffe/tree/master/examples/mnist to prepare dataset, you will get: mnist_train_lmdb, mnist_test_lmdb. Move both directories into directory `mnist_lmdb`
    ├── t10k-images-idx3-ubyte.gz       # download the following four gz file from http://yann.lecun.com/exdb/mnist/
    ├── t10k-labels-idx1-ubyte.gz
    ├── train-images-idx3-ubyte.gz
    └── train-labels-idx1-ubyte.gz

```


# AI
## Caffe

Demo Directory: caffe

Dataset Directory: `datasets/mnist/mnist_lmdb`

Program File:
```
CPU version: mnist_lenet_cpu.py
GPU version: mnist_lenet_gpu.py
```

Args available:
```
--batch_size : set batch size, default is 32
--max_steps  : set max steps for training max is 5000
```

Attention:
1. If no gpu is allocated or use cpu image to run `mnist_lenet_gpu.py`, program will crash.


## Intel-Caffe

Demo Directory: intel_caffe

Dataset Directory: `datasets/mnist/mnist_lmdb`

Program File:
```
mnist_lenet_cpu.py
```

Args available:
```
--batch_size : set batch size, default is 32
--max_steps  : set max steps for training max is 5000
```


## TensorFlow

Demo Directory: tensorflow

Dataset Directory: `datasets/cifar10/`

Program File:
```
Single node: cifar10_lenet_single.py
Multinode  : cifar10_lenet_multi_machine.py
```

Args available:
```
--batch_size : set batch size, default is 32
--max_steps  : set max steps for training max is 5000
```

Attention:
1. The log file and parameter file will be saved in train_dir in working directory, so please don't run multi tensorflow demo file in one directory, if you need run multi tensorflow demo job at the same time, please choose different working directory.
2. If you get `PermissionDeniedError: train_dir/events.out.tfevents.1525844818.compute; Permission denied`. Please check if you have write permission for the train dir.


## Tensorflow 2.0

Demo Directory: tensorflow2

Dataset directory: `datasets/mnist/mnist.npz`

Program File:
```
Single node            : mnist_single.py
Single node multi gpus : mnist_mirror_strategy.py
Multinode multi workers: mnist_multi_worker_strategy.py
Horovod distributed    : horovod_keras_mninst.py
```

Args available:
```
--train_dir or -td : the folder for seting log and saving params file
--batch_size or -b : set batch size,default is 64
--epochs or -e     : set epochs for training. default is 10
```

Attention:
1. The log file and parameter file will be saved in train_dir in working directory, so please don't run multi tensorflow demo file in one directory, if you need run multi tensorflow demo job at the same time, please choose different working directory.


## Intel-Tensorflow 2.0

Demo Directory: intel-tensorflow2

Dataset Directory: prepare as Attention described

Program File:
```
benchmarks/launch_benchmark.py
```

Args available:

training:
```
--data-location : Dataset directory.
--model-name    : set model name. use resnet50v1_5.
--framework     : set framework. use tensorflow.
--precision     : set precision. can use fp32\bfloat16.
--mode          : set mode. use training.
--batch-size    : set batch size, default is 32.
--steps         : set steps. default is 50.
--output-dir    : output dir. default is /benchmarks/common/tensorflow/logs.
--checkpoint    : set checkpoint dir. default is /tmp/.
--train_epochs  : set training epochs. use 1.

```

examples of training parameters：
```
--data-location /home/ILSVRC2012_imagenet/tf_records/ 
--model-name=resnet50v1_5 
--framework tensorflow 
--precision=bfloat16 
--mode=training 
--batch-size=256 
--steps=100 
--checkpoint /home/workdir/checkpoint_dir/ 
--output-dir /home/workdir/output_dir 
--train_epochs=1
```

inference:
```
--data-location : Dataset directory.
--in-graph      : set pre-trained model file path.
--model-name    : set model name. use resnet50v1_5.
--framework     : set framework. use tensorflow.
--precision     : set precision. can use fp32\bfloat16\int8.
--mode          : set mode. use inference.
--batch-size    : set batch size, default is 32.
--steps         : set steps. default is 50.
--output-dir    : output dir. default is benchmarks/common/tensorflow/logs.

```
examples of inference parameters：
```
--data-location /home/ILSVRC2012_imagenet/tf_records/ 
--in-graph /home/demo/ai/intel_tensorflow2/benchmarks/resnet50_v1_5_bfloat16.pb 
--model-name resnet50v1_5 
--framework tensorflow 
--precision bfloat16 
--mode inference 
--batch-size=64 
--steps=60 
--output-dir /home/output_dir/
```

Attention:
1. The log file and parameter file will be saved in checkpoint dir in working directory, so please don't run multi tensorflow demo file in one directory, if you need run multi tensorflow demo job at the same time, please choose different working directory.
2. The log file for program execution will be placed in the folder specified by `--output-dir`
3. For inference ,need download the pre-trained model.(Select the pre training model corresponding to the set precision)
    - Resnet50v1.5 int8     : wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50v1_5_int8_pretrained_model.pb
    - Resnet50v1.5 bfloat16 : wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_v1_5_bfloat16.pb
    - Resnet50v1.5 fp32     : wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
4. Please refer to https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md Download and preprocess ImageNet dataset instructions to prepare the dataset.
The format of the processed dataset should be as follows:
```
    $ ls -1 < IMAGENET DIR >/tf_records/
    train-00000-of-01024
    train-00001-of-01024
    train-00002-of-01024
    train-00003-of-01024
    ...
    validation-00000-of-00128
    validation-00001-of-00128
    validation-00002-of-00128
    validation-00003-of-00128
    ...
```
The final dataset directory is `<IMAGENET DIR>/tf_records/`.


## Mxnet

Demo Directory: mxnet

Dataset Directory: `datasets/mnist/` (all `*.gz` files)

Program File:
```
Single node CPU: mnist_lenet.py
Single node gpu: mnist_lenet.py
Multinode      : mnist_lenet.py (--kv-store args must be passed)
```

Args available:
```
--batch-size : set batch size, default is 64
--num-epochs : set the most epochs for training, default value is 20.
--kv-store   : this args must be passed for multinode training, the value can be dist_async or dist_sync
--gpu_number : the number of setting gpu
```


## Neon

Demo Directory: neon

Dataset Directory: `datasets/cifar10/`

Program File:
```
Single node cpu: cifar10_lenet.py (The Mode should choose CPU)
Single node MKL: cifar10_lenet.py (The Mode should choose MKL)
```

Args available:
```
-z: set batch size, default is 64
-e: set the most epochs for training, default value is 20.
```

Attention:
1. MKL mode will running faster than cpu node


## keras

Demo Directory: keras

Dataset Directory: `datasets/cifar10/`

Program File:
```
Single node grid search: cifar10_cnn.py
```


## Chainer

Demo Directory: chainer

Dataset Directory: `datasets/mnist/` (all `*.gz` files)

Program File:
```
Single node CPU: train_mnist.py
Single node GPU: train_mnist_gpu.py
Multinode      : train_mnist_multi.py
```

Args available:

1. For single node CPU
    ```
    --batchsize or -b : set batch_size, default value is 100
    --epoch or -e     : set the most epochs for training, default value is 20
    --out or -o       : set the model result output directory for training, default value is ./result
    ```

2. For single node GPU
    ```
    --batchsize or -b : set batch_size, default value is 400
    --epoch or -e     : set the most epochs for training, default value is 20
    --gpu or -g       : this is a bool args, program will use GPU for training if this parameter is set.
    --gpu_number or -n: set GPU using number, default is 1, you can set it max to 2
    --out or -o       : set the model result output directory for training, default value is ./result
    ```
3. For multi node 
    ```
    --batchsize or -b : set batch_size,default value is 100
    --epoch or -e     : set the most epochs for training, default value is 20
    --gpu or -g       : this is a bool args, program will use GPU for training if this parameter is set, default to use the cpu.
    --out or -o       : set the model result output directory for training, default value is ./result
    ```


## PyTorch

Demo Directory: pytorch

Dataset Directory: 
```
Single node: datasets/mnist/mnist.npz
Multi nodes: generated by program file automatically
```

Program File:
```
Single node: run_mnist_lenet.sh
Multi nodes: mnist_ddp.py
```

Args available:
1. mnist_lenet.py (Parameters that can be added to execute `mnist_lenet.py` in the `run_mnist_lenet.sh`)
```
--batchsize or -b : set batch size, default is 64
--epochs or -e    : set epochs for training, default value is 10.
```
2. mnist_ddp.py
```
--batch-size         : set batch size (default: 64)
--test-batch-size    : set test batch size (default: 1000)
--epochs             : set epochs for training (default: 14)
--lr                 : learning rate (default: 1.0)
--gamma              : learning rate step gamma (default: 0.7)
--no-cuda            : disables CUDA training (if not added, use cuda)
--seed               : random seed (default: 1)
--log-interval       : how many batches to wait before logging training status (default: 10)
--save-model         : For Saving the current Model (if added, save model)
--world-size         : number of distributed processes (default: 1)
--dist-url           : url used to set up distributed training (default: env://)
--dist-backend       : distributed backend (default: nccl)
--gloo-socket-ifname : gloo socket ifname (default: ib0)
--nccl-socket-ifname : nccl socket ifname (default: ib0)
```


Attention:
1. Distributed GPU training needs to set `--dist-backend=nccl` and `--nccl-socket-ifname=ens6f0`(or other network interface name)
2. Distributed CPU training needs to set `--dist-backend=gloo` , `--no-cuda` and `--gloo-socket-ifname=ib0`(or other network interface name)
3. `mnist_lenet.py` needs to be used by running `run_mnist_lenet.sh`


## Intel-PyTorch

Demo Directory: intel_pytorch

Dataset Directory: prepare as Attention described

Program File:
```
imagenet_lenet.py
```

Args available:
```
--data or -d                  : path to dataset (default: None)
--arch or -a                  : model architecture: alexnet | densenet121 | densenet161 | densenet169 | densenet201 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
--epochs                      : number of total epochs to run (default: 90)
--batch-size or -b            : set batch size (default: 256)
--learning-rate or --lr       : initial learning rate
--print-freq or -p            : print frequency (default: 10)
--pretrained                  : use pre-trained model
--resume                      : path to latest checkpoint (default: none)
--evaluate or -e              : evaluate model on validation set
--ipex                        : use intel pytorch extension
--workers or -j               : number of data loading workers (default: 4)
--start-epoch                 : manual epoch number (useful on restarts)
--momentum                    : momentum
--weight-decay or --wd        : weight decay (default: 1e-4)
--world-size                  : number of nodes for distributed training
--rank                        : node rank for distributed training
--dist-url                    : url used to set up distributed training
--dist-backend                : distributed backend
--seed                        : seed for initializing training.
--gpu                         : GPU id to use.
--multiprocessing-distributed : Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.
--no-cuda                     : disable CUDA
--precision                   : set precision of running model, can be fp32, bf16 or int8(todo)
--jit                         : enable jit fusion path
--dummy                       : using dummu data to test the performance of inference
--warmup-iterations or -w     : number of warmup iterations to run
```

Attention:
1. Please select from the http://www.image-net.org/ Download and extract the ImageNet2012 training and validation dataset, and then use the https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh Script to move the validation image to the marked subfolder.
After running the data preparation script, your folder structure should be as follows:
```
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder containing the val and train directories should be set to the dataset directory (for example: `--data=/home/<user>/imagent`).

2. If you want to train the model, you can run it as follows: `python imagenet_lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100`
3. If you want to use a trained model for inference, you can run it as follows: `python imagenet_lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100 --pretrained --evaluate`
4. If you are using Intel optimized Pytorch and wish to use IPEX for acceleration, please add the `--ipex` parameter
5. The dataset directory is `imagenet`.


## Scikit

Demo Directory: scikit

Dataset Directory: generated by program file automatically

Program File:
```
Single node: bench_plot_neighbors.py
```


## TensorFlow + Hbase

Demo Directory: tensorflow_hbase

Dataset Directory: `tensorflow_hbase/data/cnews` (cnews.train.txt, cnews.test.txt, cnews.val.txt)

Program File:
```
run_cnn.py (train|test) # For training model and testing model
predict.py              # For predict
```

Attention:
1. Prepare Dataset as text clasification format. and split dataset into three part: cnews.train.txt, cnews.test.txt, cnews.val.txt. Move these files to `tensorflow_hbase/data/cnews`
2. Dataset is stored at hbase. About how to set hbase server's host,port and dataset format, you can see `hbclient.py` for more detail.
3. After training, the model will be stored at `checkpoints/textcnn`, tensorboard file will be stored at `tensorboard/textcnn`


## TensorFlow + Mongodb

Demo Directory: tensorflow_mongo

Dataset Directory: `tensorflow_mongo/data/cnews` (cnews.train.txt, cnews.test.txt, cnews.val.txt)

Program File:
```
run_cnn.py (train|test) # For training model and testing model
predict.py              # For predict
```

Attention:
1. Prepare Dataset as text clasification format. and split dataset into three part: cnews.train.txt, cnews.test.txt, cnews.val.txt. Move these files to `tensorflow_mongo/data/cnews`
2. Dataset is stored at mongodb. About how to set mongodb server's host、port and dataset format, you can see `mgclient.py` for more detail.
3. After training, the model will be stored at `checkpoints/textcnn/`, tensorboard file will be stored at `tensorboard/textcnn`


## PaddlePaddle

Demo Directory: paddlepaddle

Dataset Directory: generated automatically

Program File:
```
Single node with one gpu                                                     : mnist.py
Singe node with multi gpus or Multi nodes with multi gpus (Collective mode)  : resnet/train_fleet_dygraph.py
Singe node with multi gpus or Multi nodes with multi gpus (PS mode)          : wide_and_deep/train_wide_and_deep.py
```

Args available:
1. mnist.py
    ```
    --batch_size or -b        : set batch_size for training, default 64
    --valid_batch_size or -vb : set batch_size for validation, default 64
    --epochs or -e            : set the most epochs for training, default 10
    --model_path or -m        : set model path, default 'output/mnist'
    ```
2. resnet/train_fleet_dygraph.py
    ```
    --learning_rate or -lr : set learning_rate, default 0.1
    --batch_size or -b     : set batch_size for training, default 32
    --epochs or -e         : set the most epochs for training, default 10
    ```
3. wide_and_deep/train_wide_and_deep.py
    ```
    --learning_rate or -lr    : set learning_rate, efault 0.0001
    --batch_size or -b        : set batch_size for training, default 16
    --infer_batch_size or -ib : set batch_size for inference, default 16
    --epochs or -e            : set the most epochs for training, default 10
    --data_path or -d         : set data path, default './data'
    ```


# HPC

## MPI

Demo Directory: hpc/MPI

Note: The compilation parameters of MPI and Intel-MPI are different, please confirm that the previously compiled files may be overwritten in the demo directory

Download the NASA MPI Benchmark test program: https://www.nas.nasa.gov/assets/npb/NPB3.4.2-MZ.tar.gz . After decompression, use the program in the `NPB3.4.2-MZ/NPB3.4-MZ-MPI` directory as the test scripts.

1. Copy the files in `hpc/MPI` to `NPB3.4.2-MZ/NPB3.4-MZ-MPI/config/` directory
2. Modify `NPB3.4.2-MZ/NPB3.4-MZ-MPI/BT-MZ/Makefile` to ensure that the corresponding makefile can be automatically used when compiling

```
# replace include ../config/make.def with the following code
include $(D)
```

Compile:
1. Compile the file before using it, specific operations are as follows (MPI Fortran is installed by default):
    ```
    cd <nasa-mpi-benchmark_download_path>/NPB3.4.2-MZ/NPB3.4-MZ-MPI/
    #NPROCS=6, The default number of job cores is customizable
    make clean
    make bt-mz CLASS=C NPROCS=6 D=../config/make.def
    ```

Program File:
```
bin/bt-mz.C.x
```

## Intel-MPI

Demo Directory: hpc/MPI

Note: The compilation parameters of Intel-MPI and MPI are different, please confirm that the previously compiled files may be overwritten in the demo directory

Download the NASA MPI Benchmark test program: https://www.nas.nasa.gov/assets/npb/NPB3.4.2-MZ.tar.gz . After decompression, use the program in the `NPB3.4.2-MZ/NPB3.4-MZ-MPI` directory as the test scripts.

1. Copy the files in `hpc/MPI` to `NPB3.4.2-MZ/NPB3.4-MZ-MPI/config/` directory
2. Modify `NPB3.4.2-MZ/NPB3.4-MZ-MPI/BT-MZ/Makefile` to ensure that the corresponding makefile can be automatically used when compiling

```
# replace include ../config/make.def with the following code
include $(D)
```

Compile:
1. Compile the file before using it, specific operations are as follows (Intel oneAPI is installed by default):
    ```
    cd <nasa-mpi-benchmark_download_path>/NPB3.4.2-MZ/NPB3.4-MZ-MPI/
    source <oneapi_install_path>/setvars.sh
    #NPROCS=6, The default number of job cores is customizable
    make clean
    make bt-mz CLASS=C NPROCS=6 D=../config/make.def.intel
    ```

Program File:
```
bin/bt-mz.C.x
```

## openMP

Demo Directory: hpc/OpenMP

Note: The compilation parameters of openMP and Intel-OpenMP are different, please confirm that the previously compiled files may be overwritten in the demo directory

Download the OpenMP Benchmark test program from github: https://github.com/LangdalP/EPCC-OpenMP-micro-benchmarks . Then add or replace the files in `hpc/openMP` to the project.

Compile:
1. Compile the file before using it, specific operations are as follows (Intel oneAPI is installed by default):
    ```
    cd <path/to/openmp_benchmark_project>
    make clean
    make D=Makefile.defs
    ```
   
Program File:
```
All compiled executables, such as"./taskbench","./syncbench","./schedbench", etc
```

Args available:
```
--outer-repetitions     :set outer-repetitions, default is 20 (It is recommended not to exceed 10000)
--test-time             :set test-time, default is 1000 microseconds (It is recommended not to exceed 100000)
--delay-time            :set delay-time (It is recommended not to exceed 10000)
--delay-length          :default auto-generated based on processor speed
```

Attention:
1. By default, you can use "./<binary_file>" run the programs. For example: `./taskbench`
2. If you want the program to run longer, you can control it by setting parameters. For example: `./taskbench --outer-repetitions 1000 --delay-time 100`
3. You can use the `make clean` command to clear the compile file.


## Intel-OpenMP

Demo Directory: hpc/OpenMP

Note: The compilation parameters of Intel-OpenMP and openMP are different, please confirm that the previously compiled files may be overwritten in the demo directory

Download the OpenMP Benchmark test program from github: https://github.com/LangdalP/EPCC-OpenMP-micro-benchmarks . Then add or replace the files in `hpc/openMP` to the project.


Compile:
1. Compile the file before using it, specific operations are as follows (Intel oneAPI is installed by default):
    ```
    cd <path/to/openmp_benchmark_project>
    source <oneapi_install_path>/setvars.sh
    make clean
    make D=Makefile.defs.intel
    ```
   
Program File:
```
All compiled executables, such as"./taskbench","./syncbench","./schedbench", etc
```

Args available:
```
--outer-repetitions     :set outer-repetitions, default is 20 (It is recommended not to exceed 10000)
--test-time             :set test-time, default is 1000 microseconds (It is recommended not to exceed 100000)
--delay-time            :set delay-time (It is recommended not to exceed 10000)
--delay-length          :default auto-generated based on processor speed
```

Attention:
1. By default, you can use "./<binary_file>" run the programs. For example: `./taskbench`
2. If you want the program to run longer, you can control it by setting parameters. For example: `./taskbench --outer-repetitions 1000 --delay-time 100`
3. You can use the `make clean` command to clear the compile file.



# Intel_oneAPI

## Intel_GDB

Demo Directory: intel_oneapi/Intel_GDB

#### Enable Core Dump Program

Program file

```
core_dump_demo.cpp
```

Compile command

```shell
source /opt/intel/oneapi/setvars.sh
mpiicc -g core_dump_demo.cpp -o core_dump_demo
```

#### Normal GDB Program

Program file

```
vector_mpi.c
```

Compile command

```bash
source /opt/intel/oneapi/setvars.sh
mpicc -g vector_mpi.c -o vector_mpi
```

## Intel_MPI

Same as HPC Intel-MPI

## Intel_MPITune

Intel OneAPI provides sample configuration files, which can be copied to the user directory for use. A sample configuration file can be selected as configuration file in Intel_MPITune template.

Sample Config File:
```
<oneapi_install_path>/mpi/latest/etc/tune_cfg/.
```

## Intel_Modin

Demo Directory: intel_oneapi/Intel_Modin

#### Intel Distribution of Modin

Program File:

```
demo_modin_single.py
demo_modin_single_long.py
```

#### Intel Distribution of Modin Multi Node

Program File:
```
demo_modin_multi.py
demo_modin_multi_long.py
modin_multi_read_csv.py
```


## Intel-OpenMP

Same as HPC Intel-OpenMP


## Intel_PyTorch

Demo Directory: intel_oneapi/Intel_PyTorch

#### Intel CPU Demo Program

Dataset Directory: prepare as Attention described

Program file:

```
cpu/imagenet_lenet.py
```

Args available:

```
--data or -d                  : path to dataset (default: None)
--arch or -a                  : model architecture: alexnet | densenet121 | densenet161 | densenet169 | densenet201 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
--epochs                      : number of total epochs to run (default: 90)
--batch-size or -b            : set batch size (default: 256)
--learning-rate or --lr       : initial learning rate
--print-freq or -p            : print frequency (default: 10)
--pretrained                  : use pre-trained model
--resume                      : path to latest checkpoint (default: none)
--evaluate or -e              : evaluate model on validation set
--ipex                        : use intel pytorch extension
--workers or -j               : number of data loading workers (default: 4)
--start-epoch                 : manual epoch number (useful on restarts)
--momentum                    : momentum
--weight-decay or --wd        : weight decay (default: 1e-4)
--world-size                  : number of nodes for distributed training
--rank                        : node rank for distributed training
--dist-url                    : url used to set up distributed training
--dist-backend                : distributed backend
--seed                        : seed for initializing training.
--gpu                         : GPU id to use.
--multiprocessing-distributed : Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.
--no-cuda                     : disable CUDA
--precision                   : set precision of running model, can be fp32, bf16 or int8(todo)
--jit                         : enable jit fusion path
--dummy                       : using dummu data to test the performance of inference
--warmup-iterations or -w     : number of warmup iterations to run
```

Attention:
1. Please select from the http://www.image-net.org/ Download and extract the ImageNet2012 training and validation dataset, and then use the https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh Script to move the validation image to the marked subfolder.
After running the data preparation script, your folder structure should be as follows:
```
imagenet
├── train
│   ├── n02085620
│   │   ├── n02085620_10074.JPEG
│   │   ├── n02085620_10131.JPEG
│   │   ├── n02085620_10621.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   └── ...
    └── ...
```
The folder containing the val and train directories should be set to the dataset directory (for example: -- data=/home/<user>/imagent).

2. If you want to train the model, you can run it as follows: ` Python imagenet_ lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100`
3. If you want to use a trained model for inference, you can run it as follows: ` Python imagenet_ lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100 --pretrained --evaluate`
4. If you are using Intel optimized Pytorch and wish to use IPEX for acceleration, please add the '-- IPEX' parameter
5. The dataset directory is 'imagenet'.

#### Intel XPU Demo Program

Dataset Directory: `datasets/mnist/mnist.npz`

Program file:

```
xpu/mnist_lenet.py
```

Args available:

```
--batchsize or -b : set batch size, default is 64
--epochs or -e    : set epochs for training, default value is 10.
```


## Intel_Python

Demo Directory: intel_oneapi/Intel_Python

Program File:

```
numpy_code.py
skl_demo.py
```


## Intel_TensorFlow

Demo Directory: intel_oneapi/Intel_TensorFlow

Dataset Directory: `datasets/mnist/mnist.npz`

#### Intel CPU Demo Program

Program File:

```
Single node             : cpu/mnist_single_cpu.py
Multinode multi workers : cpu/mnist_multi_worker_strategy_cpu.py
horovod                 : cpu/horovod_keras_mninst.py
```

horovod run Parameters:

```
horovodrun -np 2 -H c1:2 python train.py
horovodrun -np 4 -H C1:2,C2:2 python train.py
```


#### Intel XPU Demo Program

Program File:
```
Single node                : mnist_single.py
Single node and multi-card : horovod_keras_mninst.py、mnist_mirror_strategy.py
Multi nodes and multi-card : horovod_keras_mninst.py
```

Run Parameters:
``` 
horovodrun -np 2 -H c1:2 python train.py
horovodrun -np 4 -H C1:2,C2:2 python train.py

```
