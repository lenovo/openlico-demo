# Copyright 2015-present Lenovo
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
 
N = 102400
 
x = np.linspace(0.0123, 4567.89, N)
 
def mine(x,Z,func,name):
 
  print( name)
 
  start = time.time()
 
  for z in range ( 0, Z ) :
 
    y = func(x)
 
  end = time.time()
 
  print (N, Z, end - start)
 
  return
 
mine(x,10000,np.sin,'np.sin')
 
mine(x,10000,np.cos,'np.cos')
 
mine(x,10000,np.tan,'np.tan')


