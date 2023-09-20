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
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class LocalDataset(Dataset):
    def __init__(self, base_path, train=False):
        with np.load(base_path) as f:
            if train:
                self.x, self.y = f['x_train'], f['y_train']
            else:
                self.x, self.y = f['x_test'], f['y_test']

        self.transformations = \
            transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

    def __getitem__(self, index):
        img = self.transformations(self.x[index])
        label = int(self.y[index])
        return img, label

    def __len__(self):
        return len(self.x)
