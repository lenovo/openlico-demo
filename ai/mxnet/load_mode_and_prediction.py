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

import mxnet as mx
import matplotlib.pyplot as plt
from download_train_data import val_img
import numpy as np

model = mx.model.FeedForward.load('/code/my_mxnet/mymnist_model', 100)
for i in range(3):
  plt.imshow(val_img[i], cmap='Greys_r')
  plt.axis('off')
  plt.show()
  prob = model.predict(val_img[i:i+1].astype(np.float32)/255)[0]
  assert max(prob) > 0.80, "Low prediction accuracy."
  print('Classified as %d with probability %f' % (prob.argmax(), max(prob)))
# print'Validation accuracy: %f%%'% (model.score(val_iter)*100,)