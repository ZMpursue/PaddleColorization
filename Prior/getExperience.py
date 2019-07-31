import numpy as np
import os
from skimage import color,io
import cv2
import sklearn.neighbors as neighbors

'''读取Q区间'''
Q = np.load('./Q.npy')
count = np.zeros([313],dtype=int)

def lab2Q(i,):
    '''通过最领近搜索算法（KNN-ball_tree）和高斯函数(取sigma为5)将Lab空间上的点映射到Q空间
    INPUT
        i   除去L层的Lab图像矩阵
    OUTPUT
        o   对应的Q空间上的2维数组
    '''
    array = nd_to_2d(i,axis=2)
    nbrs = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(Q)
    d,n = nbrs.kneighbors(array)
    w = np.exp(-d**2/(2*5**2))
    w = w / np.sum(w ,axis=1)[:,np.newaxis]
    (x,y) = w.shape
    o = []
    for m in range(x):
        ans = np.array([0,0])
        for t in range(y):
            ans = np.add(ans, w[m][t]*Q[n[m][t]])
        #print(ans)
        o.append(ans)
    o = np.array(o)
    return o

def distribution(i):
    '''对Q空间进行统计，得到经验分布
    INPUT
        i   Q空间上的2维数组
    OUTPUT
        None
    '''
    nbrs = neighbors.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Q)
    d, n = nbrs.kneighbors(i)
    for t in n:
        count[t[0]] = count[t[0]] + 1

def nd_to_2d(i,axis=2):
    '''将N维的矩阵转为2维矩阵
    INPUT
        i       N维矩阵
        axis    需要保留的维度
    OUTPUT
        o       转换的2维矩阵
    '''
    n = i.ndim
    shapeArray = np.array(i.shape)
    diff = np.setdiff1d(np.arange(0,n),np.array(axis))
    p = np.prod(shapeArray[diff])
    ax = np.concatenate((diff,np.array(axis).flatten()),axis = 0)
    o = i.transpose((ax))
    o = o.reshape(p,shapeArray[axis])
    return o

if __name__ == '__main__':
    path = '../data'
    files = os.listdir(path)
    n = 1
    for file in files:
        img = io.imread(path + '/' + file)
        if img is None:
            continue
        if len(img.shape) != 3:
            continue
        imgC = color.rgb2lab(img)
        imgC = imgC[:, :, 1:3]
        distribution(lab2Q(imgC))
        print('第' + str(n) + '张图片...Done!')
        n = n + 1
    exp = count/np.sum(count,axis=0)
    np.save('Experience.npy',exp)


