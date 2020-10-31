import os
import cv2
import numpy as np
import paddle.dataset as dataset
from skimage import io,color,transform
import sklearn.neighbors as neighbors

'''准备数据，定义Reader()'''

PATH = 'work/train/'
TEST = 'work/train/'


class DataGenerater:
    def __init__(self):
        datalist = os.listdir(PATH)
        self.testlist = os.listdir(TEST)
        self.datalist = datalist

    def load(self, image):
        '''读取图片,并转为Lab，并提取出L和ab'''
        img = io.imread(image)
        lab = np.array(color.rgb2lab(img)).transpose()
        l = lab[:1, :, :]
        l = l.astype('float32')
        ab = lab[1:, :, :]
        ab = ab.astype('float32')
        return l, ab

    def create_train_reader(self):
        '''给dataset定义reader'''

        def reader():
            for img in self.datalist:
                # print(img)
                try:
                    l, ab = self.load(PATH + img)
                    # print(ab)
                    yield l.astype('float32'), ab.astype('float32')
                except Exception as e:
                    print(e)

        return reader

    def create_test_reader(self, ):
        '''给test定义reader'''

        def reader():
            for img in self.testlist:
                l, ab = self.load(TEST + img)
                yield l.astype('float32'), ab.astype('float32')

        return reader


def train(batch_sizes=32):
    reader = DataGenerater().create_train_reader()
    return reader


def test():
    reader = DataGenerater().create_test_reader()
    return reader