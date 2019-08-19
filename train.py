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
from mxnet import gluon, autograd, nd, init
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata

net = nn.Sequential()
net.add(nn.Conv2D(16, (5,5), strides=(2,2), activation="relu"))
net.add(nn.Conv2D(32, (5,5), activation="relu"))
net.add(nn.Flatten())
net.add(nn.Dense(128, activation="relu"))
net.add(nn.Dense(10, activation="relu"))
net.initialize(init.Normal(sigma=0.01),ctx=mx.gpu())

loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

num_epochs = 100000

batch_size = 1000

dataset = gdata.ArrayDataset(
    nd.array(train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2]),ctx=mx.gpu(),dtype=np.float32), 
    nd.array(train_labels, ctx=mx.gpu(), dtype=np.float32))
data_iter = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(1,num_epochs+1):
    for sample, label in data_iter:
        with autograd.record():
            y = net(sample)
            l = loss(y, label.copyto(mx.gpu()))
        l.backward()
        trainer.step(batch_size)
    y = net(nd.array(train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2]), ctx=mx.gpu(), dtype=np.float32))
    l = loss(y, nd.array(train_labels, ctx=mx.gpu(), dtype=np.float32))
    print('epoch %d,loss: %f'%(epoch,l.mean().asnumpy()))