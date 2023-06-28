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

import paddle
import argparse
import numpy as np
from paddle.vision.transforms import Normalize


def main(args):
    transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
    # download and init DataSet
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

    # model networking and initialize the network
    lenet = paddle.vision.models.LeNet(num_classes=10)
    model = paddle.Model(lenet)

    #prepare configuration, loss function, optimizer, evaluation indicator
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
                  paddle.nn.CrossEntropyLoss(),
                  paddle.metric.Accuracy())

    # train model
    model.fit(train_dataset, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    # model evaluation
    model.evaluate(test_dataset, batch_size=args.valid_batch_size, verbose=1)

    # save model
    model.save(args.model_path)
    # load model
    model.load(args.model_path)

    # sample one pic from test dataset
    img, label = test_dataset[0]
    # change shape of image shape from 1*28*28 to 1*1*28*28, add batch dimention to image 
    # to fit the requirements of the model input
    img_batch = np.expand_dims(img.astype('float32'), axis=0)

    # proceed inference, out type is list, sort the inference result from out data
    out = model.predict_batch(img_batch)[0]
    pred_label = out.argmax()
    print('true label: {}, pred label: {}'.format(label[0], pred_label))
    # visualize image
    #from matplotlib import pyplot as plt
    #plt.imshow(img[0])

if __name__ == '__main__':
    # training params settings
    parser = argparse.ArgumentParser(description='PaddlePaddle 2.3.0 MNIST Example')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', '-vb', type=int, default=64,
                        help='input batch size for validing (default: 64)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model_path', '-m', type=str, default='output/mnist',
                        help='path of model (default: "output/mnist")')
    args = parser.parse_args()
    main(args)
