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

NUM = 50000000

def modin_pandas_test():
    import ray
    ray.init()  
    import modin.pandas as pd
    from time import time
    df = pd.DataFrame(zip(range(NUM),range(NUM*2,NUM*3)),columns=['a','b'])
    start = time()
    df['c'] = df.apply(lambda x:x.a*2+x.b%99 ,axis=1)
    df['d'] = df.apply(lambda x:1 if x.a%2==0 else 0, axis=1)
    df['e'] = df.apply(lambda x:x.a%67+x.b%33+x.c%666+x.d*123 ,axis=1)
    df['c1'] = df.apply(lambda x:x.a*2+x.b%99 ,axis=1)
    df['c2'] = df.apply(lambda x:x.a*2+x.b%99 ,axis=1)
    df['c3'] = df.apply(lambda x:x.a*2+x.b%99 ,axis=1)
    df['d4'] = df.apply(lambda x:1 if x.a%2==0 else 0, axis=1)
    df['e5'] = df.apply(lambda x:x.a%67+x.b%33+x.c%666+x.d*123 ,axis=1)
    df['d6'] = df.apply(lambda x:1 if x.a%2==0 else 0, axis=1)
    df['e7'] = df.apply(lambda x:x.a%67+x.b%33+x.c%666+x.d*123 ,axis=1)
    df['d8'] = df.apply(lambda x:1 if x.a%2==0 else 0, axis=1)
    df['e9'] = df.apply(lambda x:x.a%67+x.b%33+x.c%666+x.d*123 ,axis=1)
    print('modin_pandas_df.apply Time: {:5.2f}s'.format(time() - start))

if __name__ == '__main__':
    print("Modin START!")
    modin_pandas_test()
    print("Modin END!")

