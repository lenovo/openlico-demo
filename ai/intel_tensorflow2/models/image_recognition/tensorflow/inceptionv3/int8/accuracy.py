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
import os
import time
import numpy as np

from google.protobuf import text_format
import tensorflow as tf
import preprocessing
import datasets

NUM_TEST_IMAGES = 50000

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')

  return graph

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_graph", default=None,
                      help="graph/model to be executed")
  parser.add_argument("--data_location", default=None,
                      help="full path to the validation data")
  parser.add_argument("--input_height", default=299,
                      type=int, help="input height")
  parser.add_argument("--input_width", default=299,
                      type=int, help="input width")
  parser.add_argument("--batch_size", default=32,
                      type=int, help="batch size")
  parser.add_argument("--input_layer", default="input",
                      help="name of input layer")
  parser.add_argument("--output_layer", default="predict",
                      help="name of output layer")
  parser.add_argument(
      '--num_inter_threads',
      help='number threads across operators',
      type=int, default=1)
  parser.add_argument(
      '--num_intra_threads',
      help='number threads for an operator',
      type=int, default=1)
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
  num_inter_threads = args.num_inter_threads
  num_intra_threads = args.num_intra_threads
  data_location = args.data_location

  data_graph = tf.Graph()
  with data_graph.as_default():
    dataset = datasets.ImagenetData(data_location)
    preprocessor = dataset.get_image_preprocessor()(
        input_height, input_width, batch_size,
        1, # device count
        tf.float32, # data_type for input fed to the graph
        train=False, # doing inference
        resize_method='bilinear')

    images, labels = preprocessor.minibatch(dataset, subset='validation',
                      use_datasets=True, cache_data=False)

  infer_graph = load_graph(model_file)

  input_tensor = infer_graph.get_tensor_by_name(input_layer + ":0")
  output_tensor = infer_graph.get_tensor_by_name(output_layer + ":0")
  
  config = tf.compat.v1.ConfigProto()
  config.inter_op_parallelism_threads = num_inter_threads
  config.intra_op_parallelism_threads = num_intra_threads

  total_accuracy1, total_accuracy5 = (0.0, 0.0)
  num_processed_images = 0
  num_remaining_images = dataset.num_examples_per_epoch(subset='validation') \
                            - num_processed_images
  with tf.compat.v1.Session(graph=data_graph) as sess:
    sess_graph = tf.compat.v1.Session(graph=infer_graph, config=config)
    while num_remaining_images >= batch_size:
      # Reads and preprocess data
      np_images, np_labels = sess.run([images[0], labels[0]])
      num_processed_images += batch_size
      num_remaining_images -= batch_size
      start_time = time.time()
      # Compute inference on the preprocessed data
      predictions = sess_graph.run(output_tensor,
                             {input_tensor: np_images})
      elapsed_time = time.time() - start_time
      accuracy1 = tf.reduce_sum(
          input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
              targets=tf.constant(np_labels), k=1), tf.float32))

      accuracy5 = tf.reduce_sum(
          input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
              targets=tf.constant(np_labels), k=5), tf.float32))
      np_accuracy1, np_accuracy5 =  sess.run([accuracy1, accuracy5])
      total_accuracy1 += np_accuracy1
      total_accuracy5 += np_accuracy5
      print("Iteration time: %0.4f ms" % elapsed_time)
      print("Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)" \
          % (num_processed_images, total_accuracy1/num_processed_images,
          total_accuracy5/num_processed_images))
