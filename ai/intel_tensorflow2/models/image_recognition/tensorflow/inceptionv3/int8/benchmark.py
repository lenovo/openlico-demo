#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import datasets
import tensorflow as tf

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_graph", default=None,
                      help="graph/model to be executed")
  parser.add_argument("--input_height", default=299,
                      type=int, help="input height")
  parser.add_argument("--input_width", default=299,
                      type=int, help="input width")
  parser.add_argument("--batch_size", default=32,
                      type=int, help="batch size")
  parser.add_argument("--data_location", default=None,
                      help="dataset location")
  parser.add_argument("--input_layer", default="input",
                      help="name of input layer")
  parser.add_argument("--output_layer", default="predict",
                      help="name of output layer")
  parser.add_argument("--num_cores", default=28,
                      type=int, help="number of physical cores")
  parser.add_argument(
    '--num_inter_threads',
    help='number threads across operators',
    type=int, default=1)
  parser.add_argument(
    '--num_intra_threads',
    help='number threads for an operator',
    type=int, default=1)
  parser.add_argument(
    '--data_num_inter_threads',
    help='number threads across data layer operators',
    type=int, default=16)
  parser.add_argument(
    '--data_num_intra_threads',
    help='number threads for an data layer operator',
    type=int, default=14)
  parser.add_argument("--warmup_steps", type=int, default=10,
                      help="number of warmup steps")
  parser.add_argument("--steps", type=int, default=50, help="number of steps")
  args = parser.parse_args()

  if args.input_graph:
    model_file = args.input_graph
  else:
    sys.exit("Please provide a graph file.")
  if args.input_height:
    input_height = args.input_height
  else:
    input_height = 299
  if args.input_width:
    input_width = args.input_width
  else:
    input_width = 299
  batch_size = args.batch_size
  input_layer = args.input_layer
  output_layer = args.output_layer
  warmup_steps = args.warmup_steps
  steps = args.steps
  assert steps > 10, "Benchmark steps should be at least 10."
  num_inter_threads = args.num_inter_threads
  num_intra_threads = args.num_intra_threads

  data_config = tf.compat.v1.ConfigProto()
  data_config.intra_op_parallelism_threads = args.data_num_intra_threads
  data_config.inter_op_parallelism_threads = args.data_num_inter_threads
  data_config.use_per_session_threads = 1

  infer_config = tf.compat.v1.ConfigProto()
  infer_config.intra_op_parallelism_threads = num_intra_threads
  infer_config.inter_op_parallelism_threads = num_inter_threads
  infer_config.use_per_session_threads = 1

  data_graph = tf.Graph()
  with data_graph.as_default():
    if args.data_location:
      print("inference with real data")
      # get the images from dataset
      dataset = datasets.ImagenetData(args.data_location)
      preprocessor = dataset.get_image_preprocessor(benchmark=True)(
        input_height, input_width, batch_size,
        num_cores=args.num_cores,
        resize_method='bilinear')
      images = preprocessor.minibatch(dataset, subset='validation')
    else:
      # synthetic images
      print("inference with dummy data")
      input_shape = [batch_size, input_height, input_width, 3]
      images = tf.random.uniform(
        input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')

  infer_graph = tf.Graph()
  with infer_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

  input_tensor = infer_graph.get_tensor_by_name(input_layer + ":0")
  output_tensor = infer_graph.get_tensor_by_name(output_layer + ":0")
  tf.compat.v1.global_variables_initializer()

  data_sess = tf.compat.v1.Session(graph=data_graph, config=data_config)
  infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)

  print("[Running warmup steps...]")
  for t in range(warmup_steps):
    data_start_time = time.time()
    image_data = data_sess.run(images)
    data_load_time = time.time() - data_start_time

    start_time = time.time()
    infer_sess.run(output_tensor, {input_tensor: image_data})
    elapsed_time = time.time() - start_time

    # only count the data loading and processing time for real data
    if args.data_location:
      elapsed_time += data_load_time

    if ((t + 1) % 10 == 0):
      print("steps = {0}, {1} images/sec"
            "".format(t + 1, batch_size / elapsed_time))

  print("[Running benchmark steps...]")
  total_time = 0
  total_images = 0

  for t in range(steps):
    try:
      data_start_time = time.time()
      image_data = data_sess.run(images)
      data_load_time = time.time() - data_start_time

      start_time = time.time()
      infer_sess.run(output_tensor, {input_tensor: image_data})
      elapsed_time = time.time() - start_time

      # only count the data loading and processing time for real data
      if args.data_location:
        elapsed_time += data_load_time

      total_time += elapsed_time
      total_images += batch_size
      if ((t + 1) % 10 == 0):
        print("steps = {0}, {1} images/sec"
              "".format(t + 1, batch_size / elapsed_time))
    except tf.errors.OutOfRangeError:
      print("Running out of images from dataset.")
      break

  print("Average throughput for batch size {0}: {1} images/sec".format(batch_size, total_images / total_time))

