import numpy as np
import os
import sklearn.neighbors as neighbors
from skimage import color,io

'''读取经验分布文件'''
exp = np.load('./Experience.npy')

def getWeight(a = 1,lambada = 0.5):
    '''将经验分布与权重均匀混合，取倒数，进行归一化，并使得加权后期望为1
    INPUT
        a           先验校正因子
        lambada     先验混合均匀的百分比
    OUTPUT
        ans         经验分布对应的权值矩阵
    '''
    unit = np.zeros_like(exp)
    unit[exp != 0] = 1
    unit = unit/np.sum(unit)
    mix = (1 - lambada) * exp + lambada * unit
    ans = mix ** (-a)
    ans = ans / np.sum(exp * ans)
    return ans

if __name__ == '__main__':
    w = getWeight()
    print(w)
    np.save('Weight.npy',w)