import os
import cv2
import numpy as np
import paddle.dataset as dataset
from skimage import io,color,transform
import sklearn.neighbors as neighbors
import matplotlib.pyplot as plt
import paddle
'''准备数据，定义Reader()'''

PATH = '../data/train/'
TEST = '../data/test/'
Q = np.load('../Prior/Q.npy')
Weight = np.load('../Prior/Weight.npy')

class DataGenerater:
    def __init__(self):
        '''初始化'''
        self.datalist = os.listdir(PATH)
        self.testlist = os.listdir(TEST)

    def load(self, image):
        '''读取图片,并转为Lab，并提取出L和ab'''
        img = io.imread(image)
        # mini = transform.resize(img,(64,64),mode='reflect',anti_aliasing=True, anti_aliasing_sigma=None)
        lab = np.array(color.rgb2lab(img)).transpose()
        #mini = np.array(color.rgb2lab(mini)).transpose()
        l = lab[:1,:,:]
        ab = (lab[1:,:,:] + 128)
        return l,ab


    def create_train_reader(self):
        '''给dataset定义reader'''

        def reader():
            for img in self.datalist:
                #print(img)
                try:
                    l,ab = self.load(PATH + img)
                    yield np.array([l,ab])
                except Exception as e:
                    print(e)
        return reader

    def create_test_reader(self,):
        '''给test定义reader'''

        def reader():
            for img in self.testlist:
                l,ab = self.load(TEST + img)
                yield l.astype('float32'),ab.astype('float32')

        return reader

def train(batch_sizes = 32):
    reader = DataGenerater().create_train_reader()
    return reader

def test():
    reader = DataGenerater().create_test_reader()
    return reader

'''test'''
if __name__ == '__main__':
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader=train(), buf_size=1
        ),
        batch_size=128
    )
    for batch_id, data in enumerate(train_reader()):
        for i in range(4):
            l = data[i][0][0]
            ab = data[i][1]
            a = ab[0]
            a = a - 128
            b = ab[1]
            b = b - 128
            l = l[:, :, np.newaxis]
            a = a[:, :, np.newaxis].astype('float64')
            b = b[:, :, np.newaxis].astype('float64')
            lab = np.concatenate((l, a, b), axis=2)
            # img = (255 * np.clip(color.lab2rgb(lab), 0, 1)).astype('uint8')
            img = color.lab2rgb((lab))
            img = transform.rotate(img, 270)
            plt.subplot(1, 4, i + 1)
            plt.imshow(img, vmin=-1, vmax=1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(wspace=0.1,hspace=0.1)
        plt.show()
        break