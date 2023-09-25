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

import os
import math
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed.fleet as fleet
import numpy as np

from paddle.distributed.fleet.utils.ps_util import DistributedInfer


paddle.enable_static()

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class WideDeepDataset:
    def __init__(self, data_path):
        self.file_list = [os.path.join(data_path, x)
                          for x in os.listdir(data_path)]

    def line_process(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            if features[idx] == "":
                dense_feature.append(0.0)
            else:
                dense_feature.append(
                    (float(features[idx]) - cont_min_[idx - 1]) / cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append(
                [hash(str(idx) + features[idx]) % hash_dim_])
        label = [int(features[0])]
        return [dense_feature]+sparse_feature+[label]

    def __call__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    input_data = self.line_process(line)
                    yield input_data


class WideDeepLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(WideDeepLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.wide_part = paddle.nn.Linear(
            in_features=self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=1.0 / math.sqrt(self.dense_feature_dim))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs):
        # wide part
        wide_output = self.wide_part(dense_inputs)

        # deep part
        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)

        deep_output = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)
        for n_layer in self._mlp_layers:
            deep_output = n_layer(deep_output)

        prediction = paddle.add(x=wide_output, y=deep_output)
        pred = F.sigmoid(prediction)
        return pred


class WideDeepModel:
    def __init__(self, sparse_feature_number=1000001, sparse_inputs_slots=27, sparse_feature_dim=10, dense_input_dim=13, fc_sizes=[400, 400, 400]):
        self.sparse_feature_number = sparse_feature_number
        self.sparse_inputs_slots = sparse_inputs_slots
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_input_dim = dense_input_dim
        self.fc_sizes = fc_sizes

        self._metrics = {}

    def acc_metrics(self, pred, label):
        correct_cnt = paddle.static.create_global_var(
            name="right_cnt", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_accuracy = paddle.static.accuracy(input=pred, label=label)
        batch_correct = batch_cnt * batch_accuracy

        paddle.assign(correct_cnt + batch_correct, correct_cnt)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        accuracy = correct_cnt / total_cnt

        self._metrics["acc"] = {}
        self._metrics["acc"]["result"] = accuracy
        self._metrics["acc"]["state"] = {
            "total": (total_cnt, "float32"), "correct": (correct_cnt, "float32")}

    def auc_metrics(self, pred, label):
        auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] = paddle.static.auc(input=pred,
                                                                                                 label=label,
                                                                                                 num_thresholds=2**12,
                                                                                                 slide_steps=20)

        self._metrics["auc"] = {}
        self._metrics["auc"]["result"] = auc
        self._metrics["auc"]["state"] = {"stat_pos": (
            stat_pos, "int64"), "stat_neg": (stat_neg, "int64")}

    def mae_metrics(self, pred, label):
        abserr = paddle.static.create_global_var(
            name="abserr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_abserr = paddle.nn.functional.l1_loss(
            pred, label, reduction='sum')

        paddle.assign(abserr + batch_abserr, abserr)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        mae = abserr / total_cnt

        self._metrics["mae"] = {}
        self._metrics["mae"]["result"] = mae
        self._metrics["mae"]["state"] = {
            "total": (total_cnt, "float32"), "abserr": (abserr, "float32")}

    def mse_metrics(self, pred, label):
        sqrerr = paddle.static.create_global_var(
            name="sqrerr", persistable=True, dtype='float32', shape=[1], value=0)
        total_cnt = paddle.static.create_global_var(
            name="total_cnt", persistable=True, dtype='float32', shape=[1], value=0)

        batch_cnt = paddle.sum(
            paddle.full(shape=[paddle.shape(label)[0], 1], fill_value=1.0))
        batch_sqrerr = paddle.nn.functional.mse_loss(
            pred, label, reduction='sum')

        paddle.assign(sqrerr + batch_sqrerr, sqrerr)
        paddle.assign(total_cnt + batch_cnt, total_cnt)
        mse = sqrerr / total_cnt
        rmse = paddle.sqrt(mse)

        self._metrics["mse"] = {}
        self._metrics["mse"]["result"] = mse
        self._metrics["mse"]["state"] = {
            "total": (total_cnt, "float32"), "sqrerr": (sqrerr, "float32")}

        self._metrics["rmse"] = {}
        self._metrics["rmse"]["result"] = rmse
        self._metrics["rmse"]["state"] = {
            "total": (total_cnt, "float32"), "sqrerr": (sqrerr, "float32")}

    def net(self, is_train=True):
        dense_input = paddle.static.data(name="dense_input", shape=[
                                         None, self.dense_input_dim], dtype="float32")

        sparse_inputs = [
            paddle.static.data(name="C" + str(i),
                               shape=[None, 1],
                               lod_level=1,
                               dtype="int64") for i in range(1, self.sparse_inputs_slots)
        ]

        label_input = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        self.inputs = [dense_input] + sparse_inputs + [label_input]

        self.loader = paddle.io.DataLoader.from_generator(
            feed_list=self.inputs, capacity=64, iterable=False)

        wide_deep_model = WideDeepLayer(self.sparse_feature_number, self.sparse_feature_dim,
                                        self.dense_input_dim, self.sparse_inputs_slots - 1, self.fc_sizes)

        pred = wide_deep_model.forward(sparse_inputs, dense_input)
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        label_float = paddle.cast(label_input, dtype="float32")

        with paddle.utils.unique_name.guard():
            self.acc_metrics(pred, label_input)
            self.auc_metrics(predict_2d, label_input)
            self.mae_metrics(pred, label_float)
            self.mse_metrics(pred, label_float)

        # loss
        cost = paddle.nn.functional.log_loss(input=pred, label=label_float)
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost


def distributed_training(exe, train_model, train_data_path="./data", batch_size=10, epoch_num=1):
    train_data = WideDeepDataset(data_path=train_data_path)
    reader = train_model.loader.set_sample_generator(
        train_data, batch_size=batch_size, drop_last=True, places=paddle.CPUPlace())

    for epoch_id in range(epoch_num):
        reader.start()
        try:
            while True:
                loss_val = exe.run(program=paddle.static.default_main_program(),
                                   fetch_list=[train_model.cost.name])
                loss_val = np.mean(loss_val)
                print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id, loss_val))
        except paddle.common_ops_import.core.EOFException:
            reader.reset()


def distributed_infer(exe, test_model, test_data_path="./data", batch_size=10):
    test_origin_program = paddle.static.Program()
    test_startup_program = paddle.static.Program()
    place = paddle.CPUPlace()

    with paddle.static.program_guard(main_program=test_origin_program, startup_program=test_startup_program):
        with paddle.utils.unique_name.guard():
            test_model.net(is_train=False)

    dist_infer = DistributedInfer(
        main_program=test_origin_program, startup_program=test_startup_program)

    test_data = WideDeepDataset(data_path=test_data_path)
    reader = test_model.loader.set_sample_generator(
        test_data, batch_size=batch_size, drop_last=True, places=place)

    batch_idx = 0
    with paddle.static.program_guard(main_program=dist_infer.get_dist_infer_program()):
        reader.start()
        try:
            while True:

                loss_val, auc_val, acc_val, mae_val, mse_val, rmse_val = exe.run(program=paddle.static.default_main_program(),
                                                                                 fetch_list=[test_model.cost.name,
                                                                                             test_model._metrics["auc"]["result"].name,
                                                                                             test_model._metrics["acc"]["result"].name,
                                                                                             test_model._metrics["mae"]["result"].name,
                                                                                             test_model._metrics["mse"]["result"].name,
                                                                                             test_model._metrics["rmse"]["result"].name, ])

                print("TEST ---> loss: {} auc: {} acc: {} mae: {}, mse: {} rmse: {}\n".format(np.mean(loss_val),
                                                                                              np.mean(auc_val), np.mean(acc_val), np.mean(mae_val), np.mean(mse_val), np.mean(rmse_val)))

                batch_idx += 1
                if batch_idx % 5 == 0:
                    avg_loss = fleet.metrics.sum(
                        loss_val) / float(fleet.worker_num())
                    global_auc = fleet.metrics.auc(test_model._metrics["auc"]["state"]["stat_pos"][0],
                                                   test_model._metrics["auc"]["state"]["stat_neg"][0])
                    global_acc = fleet.metrics.acc(test_model._metrics["acc"]["state"]["correct"][0],
                                                   test_model._metrics["acc"]["state"]["total"][0])
                    global_mae = fleet.metrics.mae(test_model._metrics["mae"]["state"]["abserr"][0],
                                                   test_model._metrics["mae"]["state"]["total"][0])
                    global_mse = fleet.metrics.mse(test_model._metrics["mse"]["state"]["sqrerr"][0],
                                                   test_model._metrics["mse"]["state"]["total"][0])
                    global_rmse = fleet.metrics.rmse(test_model._metrics["rmse"]["state"]["sqrerr"][0],
                                                     test_model._metrics["rmse"]["state"]["total"][0])
                    print("Global Metrics ---> average loss: {} global auc: {} global acc: {} global mae: {} global mse: {} global rmse: {}\n".format(avg_loss,
                                                                                                                                                      global_auc, global_acc, global_mae, global_mse, global_rmse))

        except paddle.common_ops_import.core.EOFException:
            reader.reset()


def clear_metric_state(model, place):
    for metric_name in model._metrics:
        for _, state_var_tuple in model._metrics[metric_name]["state"].items():
            var = paddle.static.global_scope().find_var(
                state_var_tuple[0].name)
            if var is None:
                continue
            var = var.get_tensor()
            data_zeros = np.zeros(var._get_dims()).astype(state_var_tuple[1])
            var.set(data_zeros, place)


def main(args):
    fleet.init(is_collective=False)

    model = WideDeepModel()
    model.net(is_train=True)

    optimizer = paddle.optimizer.SGD(learning_rate=args.learning_rate)

    strategy = fleet.DistributedStrategy()
    strategy.a_sync = True
    optimizer = fleet.distributed_optimizer(optimizer, strategy)

    optimizer.minimize(model.cost)


    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    if fleet.is_worker():
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        exe.run(paddle.static.default_startup_program())

        fleet.init_worker()

        distributed_training(exe, model, train_data_path=args.data_path, batch_size=args.batch_size, epoch_num=args.epochs)
        clear_metric_state(model, place)
        distributed_infer(exe, model, test_data_path=args.data_path, batch_size=args.infer_batch_size)

        fleet.stop_worker()


if __name__ == '__main__':
    # training params settings
    parser = argparse.ArgumentParser(description='PaddlePaddle 2.3.0 PS mode distibuted wide_and_deep Example')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='input batch size for training (default: 16')
    parser.add_argument('--infer_batch_size', '-ib', type=int, default=16,
                        help='input batch size for inference (default: 16)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--data_path', '-d', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/data',
                        help='path of dataset (default: "./data")')
    args = parser.parse_args()
    main(args)
