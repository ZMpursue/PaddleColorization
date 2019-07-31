import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os
sys.path.append('./train')
from skimage import io,color
import fileReader

Q = np.load('../Prior/Q.npy')
weight = np.load('../Prior/Weight.npy')
BATCH_SIZE = 1
EPOCH_NUM = 2000
Params_dirname = "../model/gray2color.inference.model"

factor = 2
w_attr = fluid.ParamAttr(learning_rate=0., regularizer=fluid.regularizer.L2Decay(0.),initializer=fluid.initializer.Bilinear())

'''自定义损失函数'''
def createLoss(predict, truth,weight):
    '''均方差'''
    loss = fluid.layers.square_error_cost(predict,truth)
    cost = fluid.layers.mean(loss)
    '''交叉熵'''
    # cost = fluid.layers.softmax_with_cross_entropy(predict,truth,soft_label=True)
    # sub = fluid.layers.elementwise_sub(predict,weight)
    # abs = fluid.layers.abs(sub)
    # re = fluid.layers.reciprocal(weight)
    # loss = fluid.layers.elementwise_mul(abs, re)
    return cost

'''ResNet网络设计'''
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=True):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)


def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp


def resnetImagenet(input):

    res1 = layer_warp(basicblock, input, 64, 128, 4, 2)
    res2 = layer_warp(basicblock, res1, 128, 256, 4, 2)
    res3 = layer_warp(basicblock, res2, 256, 512, 5, 2)
    res4 = layer_warp(basicblock, res3, 512, 512, 5, 1)
    res5 = layer_warp(basicblock, res4, 512, 512, 5, 1)
    res6 = layer_warp(basicblock, res5, 512, 512, 5, 1)
    res7 = layer_warp(basicblock, res6, 512, 512, 5, 1)
    res8 = fluid.layers.conv2d_transpose(input=res7,
                                         num_filters=256,
                                         output_size=None,
                                         filter_size=2 * factor - factor % 2,
                                         padding=1,
                                         stride=factor,
                                         groups=256,
                                         param_attr=w_attr,
                                         bias_attr=False)

    conv1 = conv_bn_layer(res8, ch_out=313, filter_size=1, stride=1, padding=0)
    conv2 = conv_bn_layer(conv1, ch_out=2, filter_size=1,stride = 1, padding=0)
    conv3 = fluid.layers.conv2d_transpose(input=conv2,
                                         num_filters=2,
                                         output_size=None,
                                         filter_size=4 * factor - factor % 4,
                                         padding=1,
                                         stride=factor * 2,
                                         groups=2,
                                         param_attr=w_attr,
                                         bias_attr=False)
    predict = fluid.layers.softmax(conv3)
    return predict


def ResNettrain():
    gray = fluid.layers.data(name='gray', shape=[1, 256, 256], dtype='float32')
    w = fluid.layers.data(name='weight', shape=[1,256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 256, 256], dtype='float32')
    predict = resnetImagenet(gray)
    cost = createLoss(predict=predict,truth=truth,weight=w)
    return predict,cost



'''VGG网络设计'''
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=1,
            pool_stride=1,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 64, 2, [0.4, 0])
    conv3 = conv_block(conv2, 128, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 256, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])
    # conv6 = conv_block(conv5, 512, 3, [0.4, 0.4, 0])
    # conv7 = conv_block(conv6, 512, 3, [0.4, 0.4, 0])
    conv8 = conv_bn_layer(conv5, ch_out=313, filter_size=1, stride=1, padding=0)
    softmax = fluid.layers.softmax(input=conv8, use_cudnn=True)
    print(softmax)
    predict = conv_bn_layer(conv8, ch_out=2, filter_size=1, stride=1, padding=0)
    return predict

def VGGtrain():
    gray = fluid.layers.data(name='gray', shape=[1, 256, 256], dtype='float32')
    w = fluid.layers.data(name='weight', shape=[1, 256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 256, 256], dtype='float32')
    predict = vgg_bn_drop(gray)
    cost = createLoss(predict=predict, truth=truth, weight=w)
    return predict, cost


'''optimizer函数'''
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.5)


train_reader = paddle.batch(reader=fileReader.train(), batch_size=BATCH_SIZE)
test_reader = paddle.batch(reader=fileReader.test(), batch_size=10)

use_cuda = False
if not use_cuda:
    os.environ['CPU_NUM'] = str(6)
feed_order = ['gray', 'truth', 'weight']
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()

'''选择VGG或者ResNet网络训练'''
predict,cost = ResNettrain()

'''优化函数'''
optimizer = optimizer_program()
optimizer.minimize(cost)

exe = fluid.Executor(place)

def train_loop():
    gray = fluid.layers.data(name='gray', shape=[1, 256, 256], dtype='float32')
    w = fluid.layers.data(name='weight', shape=[1, 256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 256, 256], dtype='float32')
    feeder = fluid.DataFeeder(
        feed_list=['gray','weight','truth'], place=place)
    exe.run(star_program)

    step = 0
    for pass_id in range(EPOCH_NUM):
        for data in train_reader():
            exe.run(main_program, feed=feeder.feed(data),fetch_list=[cost])
            step += 1
            print(step)

        if Params_dirname is not None:
            fluid.io.save_inference_model(Params_dirname, ["gray"],
                                          [predict], exe)
train_loop()

