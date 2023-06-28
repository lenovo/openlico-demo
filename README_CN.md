本项目包含提交作业的脚本

- hpc 文件夹下为 HPC 作业，包括 MPI(也可以用于 Intel-MPI) 和 OpenMP(也可以用于 Intel-OpenMP)
- ai 文件夹下为 AI 作业，包含有 caffe, intel-caffe, tensorflow, mxnet, neon, pytorch, paddlepaddle 等框架
- intel_oneapi 为 Intel-OneAPI 作业，包含有 Intel-GBD, Intel-Modin 等

使用本项目内的脚本前，需要先准备好数据集(目录名称为datasets，放置在 ai, hpc 和 intel_oneapi 的同级目录中)。 datasets 参考下面的目录结构放置数据集:

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

Demo 目录: caffe

数据集目录: `datasets/mnist/mnist_lmdb`

运行文件:
```
CPU 版本: mnist_lenet_cpu.py
GPU 版本: mnist_lenet_gpu.py
```

参数介绍:
```
--batch_size : 设置 batch_size，默认为 32
--max_steps  : 设置最多训练次数，默认为 5000
```

备注:
1. 当不分配 gpu 或者使用 cpu 版本的 caffe 运行 `mnist_lenet_gpu.py`，程序会崩溃。


## Intel-Caffe

Demo 目录: intel_caffe

数据集目录: `datasets/mnist/mnist_lmdb`

运行文件:
```
mnist_lenet_cpu.py
```

参数介绍:
```
--batch_size : 设置 batch_size，默认为 32
--max_steps  : 设置最多训练次数，默认为 5000
```


## TensorFlow

Demo 目录: tensorflow

数据集目录: `datasets/cifar10/`

运行文件:
```
单机版: cifar10_lenet_single.py
多机版: cifar10_lenet_multi_machine.py
```

参数介绍:
```
--train_dir  : 设置 log 及参数文件目录，目录为工作目录下的相对路径
--batch_size : 设置 batch_size，默认为 32
--max_steps  : 设置最多训练次数，默认为 5000
```

备注:
1. 单机版会在训练开始前输出使用 CPU 还是 GPU 进行训练，多机版暂时没有此功能，可以通过查看运行时的 GPU 使用率进行判断
2. 程序运行的 log 文件及参数会放在工作目录下的 `train_dir` 中，因此不要同时在同一文件夹下运行多个 tensorflow 的 demo 程序，如果需要同时运行多个 tensorflow 的 demo 程序，可以选择不同工作目录，或者使用 `train_dir` 指定新的 log 文件夹。
3. 出现 `PermissionDeniedError: train_dir/events.out.tfevents.1525844818.compute; Permission denied` 这个错误说明 train_dir 没有写入权限，可以进后台删了这个目录


## Tensorflow2

Demo 文件目录: tensorflow2

数据集目录: `datasets/mnist/mnist.npz`

运行文件:
```
单机单卡      : mnist_single.py
单机多卡      : mnist_mirror_strategy.py
多机多 worker : mnist_multi_worker_strategy.py
```

参数介绍:

```
--train_dir or -td : 设置 log 及参数文件目录，目录为工作目录下的相对路径
--batch_size or -b : 设置 batch_size，默认为 32
--epochs or -e     : 设置模型训练轮数，默认为 10
```

备注:
1. 程序运行的 log 文件及参数会放在工作目录下的 `train_dir` 中，因此不要同时在同一文件夹下运行多个 tensorflow 的 demo 程序，如果需要同时运行多个 tensorflow 的 demo 程序，可以选择不同工作目录，或者使用 `train_dir` 指定新的 log 文件夹。


## Intel-Tensorflow 2.0

Demo 目录: intel-tensorflow2

数据集目录: 参照备注准备数据集

运行文件:
```
benchmarks/launch_benchmark.py
```

参数介绍:

训练:
```
--data-location : 数据集文件夹.
--model-name    : 设置使用的模型. 设置为 resnet50v1_5.
--framework     : 设置使用的框架. 设置为 tensorflow.
--precision     : 设置精度. 可选 fp32\bfloat16.
--mode          : 设置模式. 设置为 training.
--batch-size    : 设置 batch-size, 默认值 32.
--steps         : 设置执行步数. 默认值 50.
--output-dir    : 设置 log 输出文件夹. log 默认输出到 benchmarks/common/tensorflow/logs.
--checkpoint    : 设置 checkpoint 保存文件夹. 默认为 /tmp/.
--train_epochs  : 设置 epochs. 设置为 1.

```
训练时参数举例：
```
--data-location /home/ILSVRC2012_imagenet/tf_records/ 
--model-name=resnet50v1_5 
--framework tensorflow 
--precision=bfloat16 
--mode=training 
--batch-size=256 
--steps=100 
--checkpoint /home/workdir/checkpoint_dir/ 
--output-dir /home/workdir/ 
--train_epochs=1
```

推理:
```
--data-location : 数据集文件夹.
--in-graph      : 设置预训练模型文件路径.
--model-name    : 设置使用的模型. 设置为 resnet50v1_5.
--framework     : 设置使用的框架. 设置为 tensorflow.
--precision     : 设置精度. 可选 fp32\bfloat16\int8.
--mode          : 设置模式. 设置为 inference.
--batch-size    : 设置 batch-size, 默认值 32.
--steps         : 设置执行步数. 默认值 50.
--output-dir    : 设置 log 输出文件. log 默认输出到 benchmarks/common/tensorflow/logs.

```
推理时参数举例：
```
--data-location /home/ILSVRC2012_imagenet/tf_records/ 
--in-graph /home/demo/ai/intel_tensorflow2/benchmarks/resnet50_v1_5_bfloat16.pb 
--model-name resnet50v1_5 
--framework tensorflow 
--precision bfloat16 
--mode inference 
--batch-size=64 
--steps=60 
--output-dir /home/workdir/
```

备注:
1. 程序运行的模型及参数文件会放在 `--checkpoint` 所指定的文件夹中，因此不要同时在同一文件夹下运行多个 tensorflow 的 demo 程序，如果需要同时运行多个 tensorflow 的 demo 程序，可以指定不同文件夹。
2. 程序运行的 log 文件会放在 `--output-dir` 所指定的文件夹中。
3. 对于推理，需要下载预训练模型文件 (要选择和所设置精度对应的预训练模型)
    - Resnet50v1.5 int8     : wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50v1_5_int8_pretrained_model.pb
    - Resnet50v1.5 bfloat16 : wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50_v1_5_bfloat16.pb
    - Resnet50v1.5 fp32     : wget https://zenodo.org/record/2535873/files/resnet50_v1.pb

4. 请参阅 https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md 下载并预处理 ImageNet 数据集说明以准备数据集。
处理后的数据集格式应如下所示:
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

最终数据集目录为 `<IMAGENET DIR>/tf_records/`。


## Mxnet

Demo 目录: mxnet

数据集目录: `datasets/mnist/` (所有 `*.gz` 文件)

运行文件:
```
单机CPU版本: mnist_lenet.py
单机GPU版本: mnist_lenet.py
多机版本   : mnist_lenet.py (必须指定 --kv-store 参数，参考参数介绍部分)
```

参数介绍:
```
--batch-size  : 设置 batch_size，默认值为64
--num-epochs  : 设置最多训练 epoch 数，默认值为 20
--kv-store    : 分布式训练时必须指定此参数，可选择 dist_async(异步更新) 或者 dist_sync(同步更新)
--gpu_number  : 设置 gpu 数量
```

备注:
1. log 部分会显示程序执行的具体参数，如 `gpus=None` 表示未使用 gpu 运行，`gpus='0, 1'` 则表示使用 gpu0 和 gpu1 运行。 需要检查下是否以分配的 gpu 资源正确运行。


## Neon

Demo 目录: neon

数据集目录: `datasets/cifar10/`

运行文件:
```
单机 CPU 版本: cifar10_lenet.py，需要在 Mode 处选择 CPU
单机 MKL 版本: cifar10_lenet.py，需要在 Mode 处选择 MKL
```

参数介绍:
```
-z: 设置 batch_size，默认值为 64
-e: 设置最多训练 epoch 数，默认值为 10
```

备注:
1. MKL 模式下运行速度比 CPU 模式快
2. 数据集目录为 `datasets/cifar10/`


## keras

Demo 目录: keras

数据集目录: `datasets/cifar10/`

运行文件:
```
单节点 grid search 版本: cifar10_cnn.py
```


## Chainer

Demo 目录: chainer

数据集目录: `datasets/mnist/` (所有 `*.gz` 文件)

运行文件:
```
单机 CPU 版本        : train_mnist.py
单机 GPU 版本        : train_mnist_gpu.py
多机 CPU 和 GPU 版本 : train_mnist_multi.py
```

参数介绍:
1. 单机CPU版本
    ```
    --batch_size 或 -b  : 设置 batch_size，默认值为 100
    --epoch 或 -e       : 设置模型训练轮数，默认值为 20
    --out 或 -o         : 设置模型结果输出的位置，默认为 ./result
    --unit 或 -u        : 设置模型每层神经元的个数，默认为 1000
    ```
2. 单机GPU版本
    ```
    --batch_size 或 -b  : 设置 batch_size，默认值为 100
    --epoch 或 -e       : 设置模型训练轮数，默认值为 20
    --gpu 或 -g         : 设置使用 GPU 进行训练，运行时需要传入
    --gpu_number 或 -n  : 设置 GPU 使用个数，默认值为 1，最多为 2
    --out 或 -o         : 设置模型结果输出的位置，默认为 ./result
    --unit 或 -u        : 设置模型每层神经元的个数，默认为 1000
    ```
3. 多机CPU和GPU版本
    ```
    --batch_size 或 -b  : 设置 batch_size，默认值为 100
    --epoch 或 -e       : 设置模型训练轮数，默认值为 20
    --gpu 或 -g         : 设置使用GPU进行训练，默认为使用 CPU 进行训练
    --out 或 -o         : 设置模型结果输出的位置，默认为 ./result
    --unit 或 -u        : 设置模型每层神经元的个数，默认为 1000
    ```


## PyTorch

Demo 目录: pytorch

数据集目录:
```
单节点: datasets/mnist/mnist.npz
多节点: 运行文件自动生成
```

运行文件:
```
单节点: run_mnist_lenet.sh
多节点: mnist_ddp.py
```

参数介绍:
1. mnist_lenet.py (在 `run_mnist_lenet.sh` 脚本中执行 `mnist_lenet.py` 可以添加的参数)
```
--batchsize or -b : 设置每轮 batch 大小，默认值为 64.
--epochs or -e    : 设置训练的轮数，默认值是 10.
```
2. mnist_ddp.py
```
--batch-size         : 设置每轮 batch 大小 (默认值: 64)
--test-batch-size    : 设置测试轮 batch 大小 (默认值: 1000)
--epochs             : 设置训练的轮数 (默认值: 14)
--lr                 : 学习率 (默认: 1.0)
--gamma              : 学习率 gamma (默认: 0.7)
--no-cuda            : 禁用 CUDA (不添加则使用)
--seed               : 随机因子  (默认: 1)
--log-interval       : 多少个 batch 记录一次训练状态 (默认: 10)
--save-model         : 是否保存当前模型 (不添加则不保存)
--world-size         : 进程数  (默认: 1)
--dist-url           : 指定如何初始化进程组的 URL  (默认: env://)
--dist-backend       : 使用的 backend  (默认: nccl)
--gloo-socket-ifname : gloo socket 网络接口名称 (default: ib0)
--nccl-socket-ifname : nccl socket 网络接口名称 (default: ib0)
```


注意:
1. 分布式 GPU 训练需要传入 `--dist-backend=nccl` 和 `--nccl-socket-ifname=ens6f0`(或其他网络接口名称)
2. 分布式 CPU 训练需要传入 `--dist-backend=gloo` , `--no-cuda` 和 `--gloo-socket-ifname=ib0`(或其他网络接口名称)
3. `mnist_lenet.py` 需要通过 `run_mnist_lenet.sh` 执行


## Intel-PyTorch

Demo 目录: intel_pytorch

数据集目录: 参照备注准备数据集

运行文件:
```
imagenet_lenet.py
```

参数介绍:
```
--data or -d                  : 数据集的路径(默认:无)
--arch or -a                  : 模型架构: alexnet | densenet121 | densenet161 | densenet169 | densenet201 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | wide_resnet101_2 | wide_resnet50_2 (默认:resnet18)
--epochs                      : 要运行的epoch数(默认:90)
--batch-size or -b            : 设置batch_size(默认:256)
--learning-rate or --lr       : 最初的学习速率
--print-freq or -p            : 打印频率(默认:10)
--pretrained                  : 使用训练好的模型
--resume                      : 到最新检查点的路径(默认:无)
--evaluate or -e              : 在验证集上评估模型
--ipex                        : 使用Intel pytorch扩展
--workers or -j               : 数据加载worker的数量(默认值:4)
--start-epoch                 : 手动的epoch数(在重启时有用)
--momentum                    : 动力
--weight-decay or --wd        : weight衰减(默认:1e-4)
--world-size                  : 分布式训练节点数
--rank                        : 分布式训练的节点秩
--dist-url                    : url用于设置分布式训练
--dist-backend                : 分布式的后端
--seed                        : 初始化训练的种子
--gpu                         : 要使用的GPU id
--multiprocessing-distributed : 使用多处理分布式训练，每个节点启动N个进程，每个节点有N个gpu。这是使用PyTorch进行单节点或多节点数据并行训练的最快方式。
--no-cuda                     : 禁用CUDA
--precision                   : 设置运行模型的精度，可以是fp32, bf16或int8(todo)
--jit                         : 启用jit融合路径
--dummy                       : 用dummu数据检验推理的性能
--warmup-iterations or -w     : 要运行的预热迭代数
```

注意:
1. 请从 http://www.image-net.org/ 下载并提取 ImageNet2012 训练和验证数据集，然后使用 https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh 脚本，将验证图像移动到标记的子文件夹。
运行数据准备脚本后，您的文件夹结构应如下所示：
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
包含 val 和 train 目录的文件夹应设置为数据集目录（例如： `--data=/home/<user>/imagenet`）。

2. 如果你想训练模型，你可以这样运行它: `python imagenet_lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100`
3. 如果您想使用训练好的模型进行推断，您可以这样运行它:`python imagenet_lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100 --pretrained --evaluate`
4. 如果您正在使用 Intel 优化的 pytorch，并希望使用 IPEX 来加速，请添加 `--ipex` 参数
5. 数据集目录是 `imagenet`


## Scikit

Demo 目录: scikit

数据集目录: 运行文件自动生成

运行文件:
```
单节点: bench_plot_neighbors.py
```


## TensorFlow + Hbase

Demo 目录: tensorflow_hbase

数据集目录: `tensorflow_hbase/data/cnews` (cnews.train.txt, cnews.test.txt, cnews.val.txt)

运行文件:
```
python run_cnn.py (train|test) # 进行训练模型、测试模型
python predict.py              # 进行预测，检验模型是否准确
```

参数介绍: 无

备注:
1. 数据集可以参照文本分类的数据集格式准备，将数据集分成三部分：cnews.train.txt, cnews.test.txt, cnews.val.txt。 把这些文件放到 `tensorflow_hbase/data/cnews` 目录下
2. 数据集存放于 hbase 中，hbase server 的 host、 port 以及数据格式可参考 `hbclient.py`
3. 训练完成后，模型存放于 `checkpoints/textcnn/`，tensorboard 文件存放于 `tensorboard/textcnn`


## TensorFlow + Mongodb

Demo 目录: tensorflow_mongo

数据集目录: `tensorflow_mongo/data/cnews` (cnews.train.txt, cnews.test.txt, cnews.val.txt)

运行文件:
```
python run_cnn.py (train|test) # 进行训练模型、测试模型
python predict.py              # 进行预测，检验模型是否准确
```

参数介绍: 无

备注:
1. 数据集可以参照文本分类的数据集格式准备，将数据集分成三部分：cnews.train.txt, cnews.test.txt, cnews.val.txt。 把这些文件放到 `tensorflow_mongo/data/cnews` 目录下
2. 数据集存放于 mongodb 中，mongodb server 的 host、 port 以及数据格式可参考 `mgclient.py`
3. 训练完成后，模型存放于 `checkpoints/textcnn/`，tensorboard 文件存放于 `tensorboard/textcnn`


## PaddlePaddle

Demo 目录: paddlepaddle

数据集目录: 自动生成

运行文件:
```
单机单卡                         : mnist.py
单机多卡 or 多机多卡(collective)  : resnet/train_fleet_dygraph.py
单机多卡 or 多机多卡(PS)          : wide_and_deep/train_wide_and_deep.py
```

参数介绍:
1. mnist.py
    ```
    --batch_size 或 -b        : 设置训练的 batch_size，默认值为 64
    --valid_batch_size 或 -vb : 设置验证的 valid_batch_size，默认值为 64
    --epochs 或 -e            : 设置模型训练轮数，默认值为 10
    --model_path 或 -m        : 设置模型结果输出的位置，默认为 'output/mnist'
    ```
2. resnet/train_fleet_dygraph.py
    ```
    --learning_rate 或 -lr : 设置 learning_rate，默认值为 0.1
    --batch_size 或 -b     : 设置训练的 batch_size，默认值为 32
    --epochs 或 -e         : 设置模型训练轮数，默认值为 10
    ```
3. wide_and_deep/train_wide_and_deep.py
    ```
    --learning_rate 或 -lr    : 设置 learning_rate，默认值为 0.0001
    --batch_size 或 -b        : 设置训练的 batch_size，默认值为 16
    --infer_batch_size 或 -ib : 设置推理的 batch_size，默认值为 16
    --epochs 或 -e            : 设置模型训练轮数，默认值为 10
    --data_path 或 -d         : 设置使用数据集的路径，默认为 './data'
    ```


# HPC

## MPI

Demo 目录:  hpc/MPI

注意: MPI 与 Intel-MPI 的编译参数等不同，注意确认 demo 目录中可能会覆盖之前编译的文件

下载 NASA MPI Benchmark 测试程序: https://www.nas.nasa.gov/assets/npb/NPB3.4.2-MZ.tar.gz 。 解压后，使用 `NPB3.4.2-MZ/NPB3.4-MZ-MPI` 目录下的程序作为测试代码

1. 将 `hpc/MPI` 中的文件复制到 `NPB3.4.2-MZ/NPB3.4-MZ-MPI/config/` 目录下
2. 修改 `NPB3.4.2-MZ/NPB3.4-MZ-MPI/BT-MZ/Makefile`, 以确保编译时可以自动使用对应的 makefile

```
# 将 include ../config/make.def 改为如下代码
include $(D)
```

编译:
1. 在使用之前需要进行编译，具体操作如下（默认以安装 MPI、Fortran）:
    ```
    cd <nasa-mpi-benchmark_download_path>/NPB3.4.2-MZ/NPB3.4-MZ-MPI/
    #NPROCS=6, The default number of job cores is customizable
    make clean
    make bt-mz CLASS=C NPROCS=6 D=../config/make.def
    ```

运行文件:
```
bin/bt-mz.C.x
```

## Intel-MPI

Demo 目录: hpc/MPI

注意: Intel-MPI 与 MPI 的编译参数等不同，注意确认 demo 目录中可能会覆盖之前编译的文件

下载 NASA MPI Benchmark 测试程序: https://www.nas.nasa.gov/assets/npb/NPB3.4.2-MZ.tar.gz 。 解压后，使用 `NPB3.4.2-MZ/NPB3.4-MZ-MPI` 目录下的程序作为测试代码

1. 将 `hpc/MPI` 中的文件复制到 `NPB3.4.2-MZ/NPB3.4-MZ-MPI/config/` 目录下
2. 修改 `NPB3.4.2-MZ/NPB3.4-MZ-MPI/BT-MZ/Makefile`, 以确保编译时可以自动使用对应的 makefile

```
# 将 include ../config/make.def 改为如下代码
include $(D)
```


编译:
1. 在使用之前需要进行编译，具体操作如下（默认已安装 Intel oneAPI）:
    ```
    cd <nasa-mpi-benchmark_download_path>/NPB3.4.2-MZ/NPB3.4-MZ-MPI/
    source <oneapi_install_path>/setvars.sh
    #NPROCS=6, The default number of job cores is customizable
    make clean
    make bt-mz CLASS=C NPROCS=6 D=../config/make.def.intel
    ```

运行文件:
```
bin/bt-mz.C.x
```


## openMP

Demo 目录: hpc/OpenMP

注意: openMP 与 Intel-OpenMP 的编译参数等不同，注意确认 demo 目录中可能会覆盖之前编译的文件

在 github 下载 OpenMP Benchmark 测试程序: https://github.com/LangdalP/EPCC-OpenMP-micro-benchmarks 。 然后把 `hpc/openMP` 中的文件添加或替换到工程中。

编译:
1. 在使用之前需要进行编译，具体操作如下（默认已安装 openMP）:
    ```
    cd <path/to/openmp_benchmark_project>
    make clean
    make D=Makefile.defs
    ```
   
运行文件:
```
所有已编译的可执行文件，例如 "./taskbench","./syncbench","./schedbench", 等
```

参数介绍:
```
--outer-repetitions     ：设置outer-repetitions，默认值为20(建议不要大于10000)
--test-time             ：设置test-time，默认值为1000微秒(建议不要大于100000)
--delay-time            ：设置delay-time(建议不要大于100)
--delay-length          ：根据处理器速度自动生成
```

注意:
1. 默认情况下，你可以直接通过"./<binary_file>"运行程序，例如:`./taskbench`
2. 如果你想让程序跑的久一点，可以通过设置参数来控制，例如:`./taskbench --outer-repetitions 1000 --delay-time 100`
3. 你可以使用`make clean`命令清空编译文件。


## Intel-OpenMP

Demo 目录: hpc/OpenMP

注意: Intel-OpenMP 与 OpenMP 的编译参数等不同，注意确认 demo 目录中可能会覆盖之前编译的文件


在 github 下载 OpenMP Benchmark 测试程序: https://github.com/LangdalP/EPCC-OpenMP-micro-benchmarks 。 然后把 `hpc/openMP` 中的文件添加或替换到工程中。


编译:
1. 在使用之前需要进行编译，具体操作如下（默认已安装 Intel oneAPI）:
    ```
    cd <path/to/openmp_benchmark_project>
    source <oneapi_install_path>/setvars.sh
    make clean
    make D=Makefile.defs.intel
    ```
   
运行文件:
```
所有已编译的可执行文件，例如 "./taskbench","./syncbench","./schedbench", 等
```

参数介绍:
```
--outer-repetitions     ：设置outer-repetitions，默认值为20(建议不要大于10000)
--test-time             ：设置test-time，默认值为1000微秒(建议不要大于100000)
--delay-time            ：设置delay-time(建议不要大于100)
--delay-length          ：根据处理器速度自动生成
```

注意:
1. 默认情况下，你可以直接通过"./<binary_file>"运行程序，例如:`./taskbench`
2. 如果你想让程序跑的久一点，可以通过设置参数来控制，例如:`./taskbench --outer-repetitions 1000 --delay-time 100`
3. 你可以使用`make clean`命令清空编译文件。



# Intel_oneAPI

## Intel_GDB

Demo 目录: intel_oneapi/Intel_GDB

#### Core Dump 程序

运行文件：

```
core_dump_demo.cpp
```

编译：
```shell
source /opt/intel/oneapi/setvars.sh
mpiicc -g core_dump_demo.cpp -o core_dump_demo
```

#### 常规 GDB 程序

运行文件：

```
vector_mpi.c
```

编译：

```bash
source /opt/intel/oneapi/setvars.sh
mpicc -g vector_mpi.c -o vector_mpi
```

## Intel-MPI

同 HPC Intel-MPI


## Intel_MPITune

Intel OneAPI 提供了示例配置文件，可以将配置文件拷贝到用户目录下使用。 可以选择一个示例配置文件作为 Intel_MPITune 模板中的配置文件。

Sample Config File:
```
<oneapi_install_path>/mpi/latest/etc/tune_cfg/.
```


## Intel_Modin

Demo 目录: intel_oneapi/Intel_Modin

#### Intel Distribution of Modin

运行文件:

```
demo_modin_single.py
demo_modin_single_long.py
```

#### Intel Distribution of Modin Multi Node

运行文件:
```
demo_modin_multi.py
demo_modin_multi_long.py
modin_multi_read_csv.py
```


## Intel-OpenMP

同 HPC Intel-OpenMP


## Intel_PyTorch

Demo 目录: intel_oneapi/Intel_PyTorch

#### Intel CPU Demo Program

数据集目录: 参照备注准备数据集

运行文件:

```
cpu/imagenet_lenet.py
```

参数介绍:

```
--data或-d：数据集的路径（默认值：无）
--arch或-a：模型架构：alexnet | densent121 | densene161 | densent269 | densene201 | googlenet | inception_v3 | mnasnet0_5 | mnasnet 0_75 | mnasnet1_0 | mnaset1_3 | mobilenet_v2 | resnet101 | resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d | resnext50_32x8dd | resnext500_32x4d | shuffenet_v2x0_5| shuffnet_v2_x1_0 | shullenet_v2_x1_5 | shuffenet_v2_x2_0 |screzenet1_0|screzenet1_1|vgg11 |vgg11_bn |vgg13 |vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | wide_rsnet101_2 | wide_resnet50_2（默认值：resnet18）
--epochs                      : 要运行的总epochs数量 (默认: 90)
--batch-size or -b            : 设置批处理大小 (默认: 256)
--learning-rate or --lr       : 初始学习率
--print-freq or -p            : 打印频率 (默认: 10)
--pretrained                  : 使用预训练模型
--resume                      : 最新检查点路径 (默认: none)
--evaluate or -e              : 在验证集上评估模型
--ipex                        : 使用intel pytorch扩展
--workers or -j               : 数据加载worker的数量 (默认: 4)
--start-epoch                 : 手动epoch数 (在重新启动时有用)
--momentum                    : 动量
--weight-decay or --wd        : 权重衰减 (默认: 1e-4)
--world-size                  : 分布式训练节点数
--rank                        : 分布式训练的节点进程
--dist-url                    : 用于设置分布式训练的url
--dist-backend                : 分布式后端
--seed                        : 初始化训练的seed
--gpu                         : 要使用的GPU id
--multiprocessing-distributed : 使用多进程分布式培训为每个有N个GPU的节点启动N个进程。这是使用PyTorch进行单节点或多节点数据并行训练的最快方法。
--no-cuda                     : 禁用CUDA
--precision                   : 设置运行模型的精度，可以是fp32、bf16或int8（todo）
--jit                         : 启用jit融合的路径
--dummy                       : 使用虚拟数据测试推理性能
--warmup-iterations or -w     : 要运行的warmup迭代次数
```

注意:

1. 请从 http://www.image-net.org/ 下载并提取ImageNet2012训练和验证数据集 ，然后使用 https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh 脚本,将验证图像移动到标记的子文件夹。
运行数据准备脚本后，您的文件夹结构应如下所示：
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
包含val和train目录的文件夹应设置为 DATASET_DIR（例如：--data=/home/<user>/imagenet）。

2. 如果你想训练模型，你可以这样运行它:`python imagenet_lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100`
3. 如果您想使用训练好的模型进行推断，您可以这样运行它:`python imagenet_lenet.py --data /home/<user>/imagenet --batch-size 64 --epochs 100 --pretrained --evaluate`
4. 如果您正在使用Intel优化的pytorch，并希望使用IPEX来加速，请添加`--ipex`参数
5. 数据集目录是 `imagenet`

#### Intel XPU Demo Program

数据集目录: `datasets/mnist/mnist.npz`

运行文件:

```
xpu/mnist_lenet.py
```

参数介绍:

```
--batchsize or -b : set batch size, default is 64
--epochs or -e    : set epochs for training, default value is 10.
```

注意:

1. 数据集目录为 `datasets/mnist/mnist.npz`



## Intel_Python

Demo 目录: intel_oneapi/Intel_Python

运行文件:

```
numpy_code.py
skl_demo.py
```



## Intel_TensorFlow

Demo 目录: intel_oneapi/Intel_TensorFlow

数据集目录: `datasets/mnist/mnist.npz`

#### Intel CPU Demo Program


运行文件:

```
Single node             : cpu/mnist_single_cpu.py
Multinode multi workers : cpu/mnist_multi_worker_strategy_cpu.py
horovod                 : cpu/horovod_keras_mninst.py
```

horovod 运行参数:

```
horovodrun -np 2 -H c1:2 python train.py
horovodrun -np 4 -H C1:2,C2:2 python train.py
```

#### Intel XPU Demo Program

运行文件:
```

单机     : mnist_single.py
单机多卡 : mnist_mirror_strategy.py
多机多卡 : horovod_keras_mninst.py

```

运行参数:

``` 
horovodrun -np 2 -H c1:2 python train.py
horovodrun -np 4 -H C1:2,C2:2 python train.py

```
