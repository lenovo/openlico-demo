# Copyright 2015-2023 Lenovo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import subprocess
import os
import sklearn
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--max_steps', type=int, default = 5000)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()
max_steps=args.max_steps
batch_size= args.batch_size
path=os.path.dirname(os.path.realpath(__file__))
print path
file=path+'/lenet_caffe_train_test.prototxt'
old_str="batch_size"
file_data=""
with open(file, "r") as f:
        for line in f:
            if old_str in line:
                line = "batch_size: %d\n" %batch_size
            if "mnist_train_lmdb" in line:
                line="source: \""+path+"/../../datasets/mnist/mnist_lmdb/mnist_train_lmdb\"\n"
            if "mnist_test_lmdb" in line:
                line="source: \""+path+"/../../datasets/mnist/mnist_lmdb/mnist_test_lmdb\"\n"
            file_data += line
with open(file, "w") as f:
        f.write(file_data)

file=path+"/lenet_caffe_solver.prototxt"
old_str="max_iter"
file_data=""
with open(file, "r") as f:
        for line in f:
            if old_str in line:
                line = "max_iter: %d\n" %max_steps
            if 'net:' in line:
                 line = "net:\""+path+"/lenet_caffe_train_test.prototxt\"\n"
            file_data +=line
with open(file, "w") as f:
        f.write(file_data)
subprocess.call("caffe train --solver="+path+"/lenet_caffe_solver.prototxt",shell=True)
