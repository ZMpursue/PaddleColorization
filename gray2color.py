import paddle
import numpy as np
from skimage import io,color,transform
from paddle import fluid
import matplotlib.pyplot as plt
import os
import lab2rgb
import Prior.imageCrop as crop
import matplotlib.pyplot as plt

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
scope = fluid.core.Scope()
fluid.scope_guard(scope)
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=r'model/gray2color.inference.model',executor=exe,)

def loadImage(image):
    '''读取图片,并转为Lab，并提取出L和ab'''
    img = io.imread(image)
    lab = np.array(color.rgb2lab(img)).transpose()
    l = lab[:1, :, :]
    return l.reshape(1,1,256,256).astype('float32')


def nd_to_2d( i, axis=0):
    '''将N维的矩阵转为2维矩阵
    INPUT
        i       N维矩阵
        axis    需要保留的维度
    OUTPUT
        o       转换的2维矩阵
    '''
    n = i.ndim
    shapeArray = np.array(i.shape)
    diff = np.setdiff1d(np.arange(0, n), np.array(axis))
    p = np.prod(shapeArray[diff])
    ax = np.concatenate((diff, np.array(axis).flatten()), axis=0)
    o = i.transpose((ax))
    o = o.reshape(p, shapeArray[axis])
    return o

def run(input,output):
    '''处理图片并存储到相应位置
    INPUT
        input   输入图片路径
        output  输出图片路径
    OUTPUT
        None
    '''
    inference_transpiler_program = inference_program.clone()
    # t = fluid.transpiler.InferenceTranspiler()
    # t.transpile(inference_transpiler_program, place)
    crop.convertjpg(jpgfile=input,outdir=None)
    l = loadImage(input)
    result = exe.run(inference_program, feed={feed_target_names[0]: (l)}, fetch_list=fetch_targets)

    ab = result[0][0]
    l = l[0][0]
    img = lab2rgb.lab2rgb(l,ab)
    #img = transform.rotate(img,270)
    #img = np.fliplr(img)
    img = img.astype('float32')
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    #plt.show()
    plt.savefig(str(output),bbox_inches='tight')


if __name__ == '__main__':
    # run('/home/redflashing/PycharmProjects/基于Paddlepaddle的黑白影片色彩重建/data/train/n01514859_142.JPEG','/home/redflashing/下载/1.JPEG')
    path = './dataset/test/'
    files = os.listdir(path)
    for file in files:
        try:
            run(path+file,'output_img/'+file)
            print(file + '  done!')
        except Exception as e:
            print(e)




