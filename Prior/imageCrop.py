from PIL import Image
import os.path
import os
import threading
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''多线程将数据集中图片缩放后再裁切到512*512分辨率'''
def convertjpg(jpgfile,outdir,width=512,height=512):
    img=Image.open(jpgfile)
    (l,h) = img.size
    rate = min(l,h) / width
    if outdir == None:
        outdir = jpgfile
    else:
        outdir = os.path.join(outdir,os.path.basename(jpgfile))
    try:
        img = img.resize((int(l // rate),int(h // rate)),Image.BILINEAR)
        img = img.crop((0,0,width,height))
        img.save(outdir)
    except Exception as e:
        print(e)

class thread(threading.Thread):
    def __init__(self, threadID, path, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.path = path
        self.files = files
    def run(self):
        for file in self.files:
            convertjpg(self.path + file,self.path)

def cutArray(l, num):
  avg = len(l) / float(num)
  o = []
  last = 0.0

  while last < len(l):
    o.append(l[int(last):int(last + avg)])
    last += avg

  return o

if __name__ == '__main__':
    path = '../dataset/test/'
    files =  os.listdir(path)
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


