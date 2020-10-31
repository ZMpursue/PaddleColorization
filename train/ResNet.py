import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os
sys.path.append('./train')
from skimage import io,color
from train.fileReader import *
import matplotlib.pyplot as plt
import lab2rgb

Params_dirname = "work/model/gray2color.inference.model"
BATCH_SIZE = 30
EPOCH_NUM = 300

'''自定义损失函数'''
def createLoss(predict, truth):
    '''均方差'''
    loss1 = fluid.layers.square_error_cost(predict,truth)
    #loss2 = fluid.layers.square_error_cost(predict,fluid.layers.fill_constant(shape=[BATCH_SIZE,2,256,256],value=fluid.layers.mean(predict),dtype='float32'))
    cost = fluid.layers.mean(loss1) #+ 16.7 / fluid.layers.mean(loss2)
    return cost

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
def bn(x, name=None, act=None,momentum=0.5):
    return fluid.layers.batch_norm(
        x,
        bias_attr=None,
        # 指定偏置的属性的对象
        moving_mean_name=name + '3',
        # moving_mean的名称
        moving_variance_name=name + '4',
        # moving_variance的名称
        name=name,
        act=act,
        momentum=momentum,
    )


def resnetImagenet(input):
    #128
    x = layer_warp(basicblock, input, 64, 128, 1, 2)
    #64
    x = layer_warp(basicblock, x, 128, 256, 1, 2)
    #32
    x = layer_warp(basicblock, x, 256, 512, 1, 2)
    #16
    x = layer_warp(basicblock, x, 512, 1024, 1, 2)
    #8
    x = layer_warp(basicblock, x, 1024, 2048, 1, 2)
    #16
    x = deconv(x, num_filters=1024, filter_size=4, stride=2, padding=1)
    x = bn(x, name='bn_1', act='relu', momentum=0.5)
    #32
    x = deconv(x, num_filters=512, filter_size=4, stride=2, padding=1)
    x = bn(x, name='bn_2', act='relu', momentum=0.5)
    #64
    x = deconv(x, num_filters=256, filter_size=4, stride=2, padding=1)
    x = bn(x, name='bn_3', act='relu', momentum=0.5)
    #128
    x = deconv(x, num_filters=128, filter_size=4, stride=2, padding=1)
    x = bn(x, name='bn_4', act='relu', momentum=0.5)
    #256
    x = deconv(x, num_filters=64, filter_size=4, stride=2, padding=1)
    x = bn(x, name='bn_5', act='relu', momentum=0.5)

    x = deconv(x, num_filters=2, filter_size=3, stride=1, padding=1)
    return x


def ResNettrain():
    gray = fluid.layers.data(name='gray', shape=[1, 256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 256, 256], dtype='float32')
    predict = resnetImagenet(gray)
    cost = createLoss(predict=predict, truth=truth)
    return predict, cost


'''optimizer函数'''


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=2e-5, beta1=0.8)


train_reader = paddle.batch(paddle.reader.shuffle(
    reader=train(), buf_size=7500 * 3
), batch_size=BATCH_SIZE)
test_reader = paddle.batch(reader=test(), batch_size=10)

use_cuda = True
if not use_cuda:
    os.environ['CPU_NUM'] = str(6)
feed_order = ['gray', 'weight']
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()

'''网络训练'''
predict, cost = ResNettrain()

'''优化函数'''
optimizer = optimizer_program()
optimizer.minimize(cost)

exe = fluid.Executor(place)

plt.ion()
def train_loop():
    gray = fluid.layers.data(name='gray', shape=[1, 256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 256, 256], dtype='float32')
    feeder = fluid.DataFeeder(
        feed_list=['gray', 'truth'], place=place)
    exe.run(star_program)

    # 增量训练
    fluid.io.load_persistables(exe, 'work/model/incremental/', main_program)

    for pass_id in range(EPOCH_NUM):
        step = 0
        for data in train_reader():
            loss = exe.run(main_program, feed=feeder.feed(data), fetch_list=[cost])
            step += 1
            if step % 1000 == 0:
                try:
                    generated_img = exe.run(main_program, feed=feeder.feed(data), fetch_list=[predict])
                    plt.figure(figsize=(15, 6))
                    plt.grid(False)
                    for i in range(10):
                        ab = generated_img[0][i]
                        l = data[i][0][0]
                        a = ab[0]
                        b = ab[1]
                        l = l[:, :, np.newaxis]
                        a = a[:, :, np.newaxis].astype('float64')
                        b = b[:, :, np.newaxis].astype('float64')
                        lab = np.concatenate((l, a, b), axis=2)
                        img = color.lab2rgb((lab))
                        img = transform.rotate(img, 270)
                        img = np.fliplr(img)
                        plt.grid(False)
                        plt.subplot(2, 5, i + 1)
                        plt.imshow(img)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                    msg = 'Epoch ID={0} Batch ID={1} Loss={2}'.format(pass_id, step, loss[0][0])
                    plt.suptitle(msg, fontsize=20)
                    plt.draw()
                    plt.savefig('{}/{:04d}_{:04d}.png'.format('work/output_img', pass_id, step), bbox_inches='tight')
                    plt.pause(0.01)
                except IOError:
                    print(IOError)

                fluid.io.save_persistables(exe, 'work/model/incremental/', main_program)
                fluid.io.save_inference_model(Params_dirname, ["gray"], [predict], exe)
plt.ioff()
train_loop()