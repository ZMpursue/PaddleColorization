import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os
sys.path.append('./train')
from skimage import io,color
import train.fileReader as fileReader
import matplotlib.pyplot as plt
import lab2rgb

Q = np.load('../Prior/Q.npy')
weight = np.load('../Prior/Weight.npy')
BATCH_SIZE = 3
EPOCH_NUM = 20000
Params_dirname = "../model/gray2color.inference.model"

'''自定义损失函数'''
def createLoss(predict:paddle.fluid.Variable, truth):
    '''均方差'''
    loss1 = fluid.layers.square_error_cost(predict,truth)
    loss2 = fluid.layers.square_error_cost(predict,fluid.layers.fill_constant(shape=[BATCH_SIZE,2,512,512],value=fluid.layers.mean(predict),dtype='float32'))
    cost = fluid.layers.mean(loss1) + 5 / paddle.fluid.layers.mean(loss2)
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
    return fluid.layers.batch_norm(input=tmp,act=act,momentum=0.5)


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

###反卷积层
def deconv(x, num_filters, filter_size=5, stride=2, dilation=1, padding=2, output_size=None, act=None):
    return fluid.layers.conv2d_transpose(
        input=x,
        num_filters=num_filters,
        # 滤波器数量
        output_size=output_size,
        # 输出图片大小
        filter_size=filter_size,
        # 滤波器大小
        stride=stride,
        # 步长
        dilation=dilation,
        # 膨胀比例大小
        padding=padding,
        use_cudnn=True,
        # 是否使用cudnn内核
        act=act
        # 激活函数
    )


def resnetImagenet(input):
    res1 = layer_warp(basicblock, input, 64, 128, 1, 2)
    res2 = layer_warp(basicblock, res1, 128, 256, 1, 2)
    res3 = layer_warp(basicblock, res2, 256, 512, 4, 1)
    deconv1 = deconv(res3, num_filters=313, filter_size=4, stride=2, padding=1)
    deconv2 = deconv(deconv1, num_filters=2, filter_size=4, stride=2, padding=1)
    return deconv2


def ResNettrain():
    gray = fluid.layers.data(name='gray', shape=[1, 512,512], dtype='float32')
    #w = fluid.layers.data(name='weight', shape=[1,256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 512,512], dtype='float32')
    predict = resnetImagenet(gray)
    cost = createLoss(predict=predict,truth=truth)
    return predict,cost


'''optimizer函数'''
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=2e-5,beta1=0.8)


train_reader = paddle.batch(reader=fileReader.train(), batch_size=BATCH_SIZE)
test_reader = paddle.batch(reader=fileReader.test(), batch_size=10)

use_cuda = True
if not use_cuda:
    os.environ['CPU_NUM'] = str(6)
feed_order = ['gray', 'weight']
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()

'''网络训练'''
predict,cost = ResNettrain()

'''优化函数'''
optimizer = optimizer_program()
optimizer.minimize(cost)

exe = fluid.Executor(place)

def train_loop():
    gray = fluid.layers.data(name='gray', shape=[1, 512,512], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 512,512], dtype='float32')
    feeder = fluid.DataFeeder(
        feed_list=['gray','truth'], place=place)
    exe.run(star_program)

    step = 0
    #fluid.io.load_persistables(exe, '../model/incremental', main_program)
    for pass_id in range(EPOCH_NUM):
        for data in train_reader():
            loss = exe.run(main_program, feed=feeder.feed(data),fetch_list=[cost])
            step += 1
            if step % 10 == 0:
                print(str(loss[0]))
                generated_img = exe.run(main_program, feed=feeder.feed(data),fetch_list=[predict])
                ab = generated_img[0][0]
                l = data[0][0][0]
                img = lab2rgb.lab2rgb(l,ab)
                plt.imshow(img)
                plt.grid(False)
                plt.draw()
                plt.pause(0.01)
                fluid.io.save_inference_model(Params_dirname, ["gray"],[predict], exe)
                fluid.io.save_persistables(executor=exe,dirname='../model/incremental',main_program=main_program)
if __name__ == '__main__':
    train_loop()

