from dataloader import *

# 训练集文件
train_images_idx3_ubyte_file = 'I:/DataSet/MNIST/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'I:/DataSet/MNIST/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'I:/DataSet/MNIST/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'I:/DataSet/MNIST/t10k-labels.idx1-ubyte'

train_images = load_train_images(train_images_idx3_ubyte_file)/255
train_labels = load_train_labels(train_labels_idx1_ubyte_file)

test_images = load_test_images(test_images_idx3_ubyte_file)/255
test_labels = load_test_labels(test_labels_idx1_ubyte_file)


import mxnet as mx
import numpy as np
from mxnet import image, nd, gluon, metric as mtc, autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader

net = nn.Sequential()
net.add(nn.Conv2D(16, (5,5), strides=(2,2), activation="relu"))
net.add(nn.Conv2D(32, (5,5), activation="relu"))
net.add(nn.Flatten())
net.add(nn.Dense(128, activation="relu"))
net.add(nn.Dense(10, activation="relu"))
net.initialize(mx.initializer.Normal(sigma=0.01),ctx=mx.gpu())

cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.01})

num_epochs = 100000

batch_size = 1000

dataset = gluon.data.ArrayDataset(
    nd.array(train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2]),dtype=np.float32), 
    nd.array(train_labels, dtype=np.float32))

train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    train_loss = 0
    for samples, label in train_data:
        samples = samples.as_in_context(mx.gpu())
        labels = label.as_in_context(mx.gpu())

        losses = []
        with ag.record():
            ouputs = net(samples)
            losses = [cross_entropy(yhat, y) for yhat, y in zip(ouputs, labels)]
        for l in losses:
            ag.backward(l)
        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)
    
    print('epoch %d, train loss: %f'%(epoch, train_loss))