import numpy as np
import os
import imghdr
from PIL import Image
import threading


'''多线程将数据集中单通道图删除'''
def cutArray(l, num):
  avg = len(l) / float(num)
  o = []
  last = 0.0

  while last < len(l):
    o.append(l[int(last):int(last + avg)])
    last += avg

  return o

def deleteErrorImage(path,image_dir):
    count = 0
    for file in image_dir:
        try:
            image = os.path.join(path,file)
            image_type = imghdr.what(image)
            if image_type is not 'jpeg':
                os.remove(image)
                count = count + 1

            img = np.array(Image.open(image))
            if len(img.shape) is 2:
                os.remove(image)
                count = count + 1
        except Exception as e:
            print(e)
    print('done!')
    print('已删除数量：' +  str(count))

class thread(threading.Thread):
    def __init__(self, threadID, path, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.path = path
        self.files = files
    def run(self):
        deleteErrorImage(self.path,self.files)

if __name__ == '__main__':
    path = './work/train/'
    files =  os.listdir(path)
    files = cutArray(files,8)
    t1 = threading.Thread(target=deleteErrorImage,args=(path,files[0]))
    t2 = threading.Thread(target=deleteErrorImage,args=(path,files[1]))
    t3 = threading.Thread(target=deleteErrorImage,args=(path,files[2]))
    t4 = threading.Thread(target=deleteErrorImage,args=(path,files[3]))
    t5 = threading.Thread(target=deleteErrorImage,args=(path,files[4]))
    t6 = threading.Thread(target=deleteErrorImage,args=(path,files[5]))
    t7 = threading.Thread(target=deleteErrorImage,args=(path,files[6]))
    t8 = threading.Thread(target=deleteErrorImage,args=(path,files[7]))
    threadList = []
    threadList.append(t1)
    threadList.append(t2)
    threadList.append(t3)
    threadList.append(t4)
    threadList.append(t5)
    threadList.append(t6)
    threadList.append(t7)
    threadList.append(t8)
    for t in threadList:
        t.setDaemon(True)
        t.start()
        t.join()