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

import numpy as np
import time

import ray
ray.init()
import modin.pandas as pandas
import os

csv_path = "/home/hpcadmin/zbt/foo.csv"

if not os.path.exists(csv_path):
    array=np.random.randint(low=100,high=10000,size=(2**21,2**8))
    #array
    np.savetxt(csv_path, array, delimiter=",") #how to generate array

#read_csv
t0 = time.time()
pandas_df = pandas.read_csv(csv_path, names=["col{}".format(i) for i in range(256)])
pandas_time_read_csv = time.time()- t0

# Long apply function
t6 = time.time()
print(pandas_df.applymap(lambda x: x + 1))
pandas_time_applymap = time.time() - t6

# Concat
t8 = time.time()
print(pandas.concat([pandas_df, pandas_df], axis=0))
pandas_time_concat = time.time() - t8

print('stock Pandas end:')
print('pandas_time_read_csv is : {}'.format(pandas_time_read_csv))
print('pandas_time_applymap is : {}'.format(pandas_time_applymap))
print('pandas_time_concat is : {}'.format(pandas_time_concat))