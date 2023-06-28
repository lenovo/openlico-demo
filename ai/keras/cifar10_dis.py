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

from __future__ import print_function
from keras.models import Model
from tensorflow.python.ops.control_flow_ops import with_dependencies
import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cifar10
import sys
import random
# define flags
tf.app.flags.DEFINE_string("ps_hosts", "10.1.8.27:2223",
                           "comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "10.1.8.27:2222",
                           "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS
num_classes = 10
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
input_shape = x_train.shape[1:]
y_train_ca = keras.utils.to_categorical(y_train, num_classes)
y_test_ca = keras.utils.to_categorical(y_test, num_classes)
x_train_list = [x_train[i:i + 32] for i in range(0, len(x_train), 32)]
y_train_ca_list = [y_train_ca[i:i + 32] for i in range(0, len(x_train), 32)]
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(
    cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
this_index = 0


def make_model(lr):
    with tf.device(
            tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
        K.set_learning_phase(1)
        K.manual_variable_initialization(True)
        x_input = Input(shape=input_shape)
        conv = Conv2D(
            32, (3, 3), padding='same', input_shape=x_train.shape[1:])
        act = Activation('relu')
        conv2 = Conv2D(32, (3, 3))
        pol = MaxPooling2D(pool_size=(2, 2))
        conv3 = Conv2D(64, (3, 3), padding='same')
        conv4 = Conv2D(64, (3, 3))
        flat = Flatten()
        dense1 = Dense(512)
        dense2 = Dense(num_classes)
        act1 = Activation('softmax')
        x1 = conv(x_input)
        x1 = act(x1)
        x1 = conv2(x1)
        x1 = act(x1)
        x1 = pol(x1)
        x1 = conv3(x1)
        x1 = act(x1)
        x1 = conv4(x1)
        x1 = act(x1)
        x1 = pol(x1)
        x1 = flat(x1)
        x1 = dense1(x1)
        x1 = act(x1)
        x1 = dense2(x1)
        soft_output = act1(x1)
        model = myModel(inputs=[x_input], outputs=[soft_output])
        model.in1 = x_input
        model.ou1 = soft_output
        model.opt = tf.train.GradientDescentOptimizer(lr)
        model.targets = tf.placeholder(
            shape=(None, num_classes), dtype=tf.float32)
        model.loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(model.targets, soft_output))
        return model


class myModel(Model):
    def predict(self, x, batch_size=None, verbose=0, steps=None):
        result = []
        global this_index
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(FLAGS.task_index == 0)) as sess:
                self.inputs[0].graph._unsafe_unfinalize()
                saver = tf.train.Saver()
                logd = 'checkpoint%d/' % this_index
                saver.restore(sess, tf.train.latest_checkpoint(logd))
                x_input = [x[i:i + 32] for i in range(0, len(x), 32)]
                for x_train1 in x_input:
                    train_feed = {self.in1: x_train1}
                    o1 = sess.run(self.outputs[0], feed_dict=train_feed)
                    index = np.argmax(o1, axis=1)
                    #k_resu = [[ind] for ind in index]
                    result.extend(index)
        return np.array(result)

    def fit(self, *args, **kwargs):
        global this_index
        with tf.device(
                tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
            # model built
            # define train label and loss function, since it is a dirstributed system, we use global_step to record the process
            global_step = tf.contrib.framework.get_or_create_global_step()
            # define optimizer

            with tf.control_dependencies(self.updates):
                barrier = tf.no_op(name='update_barrier')
            # Compute gradients of `loss` for the variables in `var_list`. This is the first part of `minimize()`.
            with tf.control_dependencies([barrier]):
                grads = self.opt.compute_gradients(self.loss)
            # Apply gradients to variables. This is the second part of `minimize()`. It returns an `Operation` that
            # applies gradients.
            grad_update = self.opt.apply_gradients(
                grads, global_step=global_step)
            train_tensor = with_dependencies([grad_update],
                                             self.loss,
                                             name='train')
            #print(model.summary())
            #saver = tf.train.Saver()
            print("I'm going to 1 index%d" % FLAGS.task_index)
            sys.stdout.flush()
            #summary = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            #logd= "./checkpoint%d" % FLAGS.task_index
            print("I'm going to 2 index%d" % FLAGS.task_index)
            sys.stdout.flush()
            hooks = [tf.train.StopAtStepHook(last_step=10000)]
            this_index += 1
            logd = 'checkpoint%d/' % this_index
            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(FLAGS.task_index == 0),
                    checkpoint_dir=logd,
                    hooks=hooks) as sess:
                step = 0
                sess.run(init_op)
                while not sess.should_stop():
                    random_index = int(random.random() * len(x_train_list))
                    x_train1 = x_train_list[random_index]
                    y_train1 = y_train_ca_list[random_index]
                    train_feed = {self.in1: x_train1, self.targets: y_train1}
                    xxx, loss_value, step_value = sess.run(
                        [train_tensor, self.loss, global_step],
                        feed_dict=train_feed)
                    step += 1
                    if step % 100 == 0:
                        print([
                            FLAGS.task_index, step, loss_value, step_value + 1
                        ])
                    sys.stdout.flush()
                print("I'm done with index")
                print(FLAGS.task_index)
                sys.stdout.flush()
                #test_feed = {feature_p: x_purchase, feature_input: x_feature, feature_N: x_N, feature_weight: x_cart}
                #test_result = sess.run(output, test_feed)
                #print(test_result)
            #self.inputs.graph._unsafe_unfinalize()
            print("stop index%d" % FLAGS.task_index)
            sys.stdout.flush()


def main(_):
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        model = make_model(0.01)
        model.fit()
        k = model.predict(x_train[0:5000])
        re = reduce(lambda k, y: k + 1 if y[0] == y[1] else k,
                    zip(k, y_train[0:5000]), 0)
        print(re)


if __name__ == '__main__':
    tf.app.run()
