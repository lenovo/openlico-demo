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

'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import cifar10
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf
num_classes = 10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

from tensorflow.python.client import device_lib as _device_lib
local_device_protos = _device_lib.list_local_devices()
num_gpus=0
for x in local_device_protos:
    if x.device_type == 'GPU':
        num_gpus=num_gpus+1
# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
def create_model(learning_rate):
    with tf.device('/cpu:0'):
        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if num_gpus > 1:
        model = keras.utils.multi_gpu_model(model, gpus=num_gpus)
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=learning_rate, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)
x_train = x_train.astype('float32')

x_train /= 255

batch_size = [64, 128]
epochs = [2]
learning_rate = [0.1, 0.001]
param_grid = dict(
    batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
grid = GridSearchCV(
    estimator=model, param_grid=param_grid, n_jobs=1, scoring='accuracy')
grid_result = grid.fit(x_train, y_train)
# summarize results
print(
    "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
