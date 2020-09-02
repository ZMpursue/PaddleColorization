import os
import cv2
import numpy as np
import paddle.dataset as dataset
from skimage import io,color,transform
import sklearn.neighbors as neighbors


'''准备数据，定义Reader()'''

PATH = '../data/train/'
TEST = '../data/train/'
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
        ab = lab[1:,:,:]
        return l,ab

    def mask(self,ls,abs,ws):
        return ls.astype('float32'),abs.astype('float32'),ws.astype('float32')

    def distribution(self,i):
        '''对Q空间进行统计，得到经验分布'''

        nbrs = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(Q)
        d, n = nbrs.kneighbors(i)
        w = np.array([])
        X = np.array([])
        Y = np.array([])
        for i in range(n.shape[0]):
            w = np.append(w,Weight[n[i][0]])
            x,y = self.lab2Q(d[i],n[i])
            X = np.append(X, x)
            Y = np.append(Y, y)
        w = w.reshape([1,512,512])
        X = X.reshape([1,512,512])
        Y = Y.reshape([1,512,512])
        q = np.zeros([2,512,512])
        q[0] = X
        q[1] = Y
        return w,q

    def lab2Q(self,d,n):
        '''将输入np矩阵映射到Q空间上'''

        w = np.exp(-d ** 2 / (2 * 5 ** 2))
        w = w / np.sum(w)
        X = np.array([])
        Y = np.array([])
        for i in n:
            x = Q[i][0]
            y = Q[i][1]
            X = np.append(X,x)
            Y = np.append(Y,y)
        x = np.sum(X * w)
        y = np.sum(Y * w)
        return x,y

    def nd_to_2d(self,i, axis=0):
        '''将N维的矩阵转为2维矩阵'''

        n = i.ndim
        shapeArray = np.array(i.shape)
        diff = np.setdiff1d(np.arange(0, n), np.array(axis))
        p = np.prod(shapeArray[diff])
        ax = np.concatenate((diff, np.array(axis).flatten()), axis=0)
        o = i.transpose((ax))
        o = o.reshape(p, shapeArray[axis])
        return o

    def _2d_to_nd(self,i,axis=1):
        '''将2维np数组转换为N维'''

        a = i[:,:1].transpose().reshape([512])
        b = i[:,1:2].transpose().reshape([512,512])
        ab = np.zeros([2,512,512])
        ab[0] = a
        ab[1] = b
        return ab

    def weightArray(self,i):
        return self.distribution(self.nd_to_2d(i))


    def create_train_reader(self):
        '''给dataset定义reader'''

        def reader():
            for img in self.datalist:
                #print(img)
                try:
                    l, ab = self.load(PATH + img)
                    yield l.astype('float32'), ab.astype('float32')
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
    reader = DataGenerater().create_train_reader()
    print(reader)
    test = DataGenerater().create_test_reader()
    print(test)