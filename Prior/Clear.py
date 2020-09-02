import numpy as np
import os
import imghdr
from PIL import Image
import threading

'''多线程清理指定文件夹中损坏或不是JPEG格式的图片'''
def deleteErrorImage(path, files):
    count = 0
    for file in files:
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
    print('已删除数量: ' + str(count))

def cutArray(l, num):
  avg = len(l) / float(num)
  o = []
  last = 0.0

  while last < len(l):
    o.append(l[int(last):int(last + avg)])
    last += avg

  return o

class thread(threading.Thread):
    def __init__(self, threadID, path, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.path = path
        self.files = files

    def run(self):
        deleteErrorImage(self.path, self.files)

if __name__ == '__main__':
    path = '../dataset/test/'
    files = os.listdir(path)
    files = cutArray(files,8)
    T1 = thread(1, path, files[0])
    T2 = thread(2, path, files[1])
    T3 = thread(3, path, files[2])
    T4 = thread(4, path, files[3])
    T5 = thread(5, path, files[4])
    T6 = thread(6, path, files[5])
    T7 = thread(7, path, files[6])
    T8 = thread(8, path, files[7])
    T1.start()
    T2.start()
    T3.start()
    T4.start()
    T5.start()
    T6.start()
    T7.start()
    T8.start()
    T1.join()
    T2.join()
    T3.join()
    T4.join()
    T5.join()
    T6.join()
    T7.join()
    T8.join()