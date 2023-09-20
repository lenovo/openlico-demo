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

import gzip
import struct

import numpy
import six

from chainer.datasets import tuple_dataset

def make_npz(path, paths):

    x_path, y_path = paths

    with gzip.open(x_path, 'rb') as fx, gzip.open(y_path, 'rb') as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fx.read(4))
        if N != struct.unpack('>i', fy.read(4))[0]:
            raise RuntimeError('wrong pair of MNIST images and labels')
        fx.read(8)

        x = numpy.empty((N, 784), dtype=numpy.uint8)
        y = numpy.empty(N, dtype=numpy.uint8)

        for i in six.moves.range(N):
            y[i] = ord(fy.read(1))
            for j in six.moves.range(784):
                x[i, j] = ord(fx.read(1))

        numpy.savez_compressed(path, x=x, y=y)
        return {'x': x, 'y': y}

def preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype,
                     rgb_format):
    images = raw['x']

    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
        if rgb_format:
            images = numpy.broadcast_to(images,
                                        (len(images), 3) + images.shape[2:])
    elif ndim != 1:
        raise ValueError('invalid ndim for MNIST dataset')

    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images
