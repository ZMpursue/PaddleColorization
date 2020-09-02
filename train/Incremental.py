import paddle.fluid as fluid
import paddle
import numpy as np
import sys
import os
sys.path.append('./train')
from skimage import io,color
import train.fileReader as fileReader

Q = np.load('../Prior/Q.npy')
weight = np.load('../Prior/Weight.npy')
BATCH_SIZE = 1
EPOCH_NUM = 20000
Params_dirname = "../model/chicken.model.2"
place = fluid.CUDAPlace(0)

exe = fluid.Executor(place)
path = "../model/incremental"
startup_prog = fluid.default_startup_program()
exe.run(startup_prog)
fluid.io.load_persistables(dirname=path, main_program=startup_prog,executor=exe)
main_prog = fluid.default_main_program()

train_reader = paddle.batch(reader=fileReader.train(), batch_size=BATCH_SIZE)

def train_loop():
    gray = fluid.layers.data(name='gray', shape=[1, 256, 256], dtype='float32')
    #w = fluid.layers.data(name='weight', shape=[1, 256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 256, 256], dtype='float32')
    feeder = fluid.DataFeeder(
        feed_list=['gray','truth'], place=place)

    step = 0
    for pass_id in range(EPOCH_NUM):
        for data in train_reader():
            exe.run(main_prog, feed=feeder.feed(data))
            step += 1
            if step % 1000 == 0:
                print(step)
            # if step % 50000 == 0:
                fluid.io.save_persistables(executor=exe,dirname='../model/incremental',main_program=main_prog)
train_loop()