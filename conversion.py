import cv2
from cv2 import VideoWriter,VideoWriter_fourcc
import os
import shutil
import subprocess

'''本脚本需要配合命令行工具ffmpeg'''

def video2photo(videoName = './movie/input.mp4'):
    '''将视频转换为图片，视频默认读取路径./movie/input.mp4，图片写入路径./cache/input_img/*（从1开始给每一帧图片命名*.jpg)
    INPUTS
        videoName    视频读取路径
    OUTPUTS
        None
    '''
    video = cv2.VideoCapture(videoName)
    n = 1
    #清空图片缓存
    shutil.rmtree('./cache/input_img')
    os.mkdir('./cache/input_img')
    #清空音频缓存
    shutil.rmtree('./cache/audio')
    os.mkdir('./cache/audio')
    # 提取音频保存到./cache/audio/
    # 运行终端程序提取音频
    command = 'ffmpeg -i ' + videoName + ' -ab 160k -ac 2 -ar 44100 -vn ./cache/audio/audio.wav'
    subprocess.call(command, shell=True,stdout= open('/dev/null','w'),stderr=subprocess.STDOUT)

    if video.isOpened():
        rval, frame = video.read()
    else:
        rval = False
    while rval:
        try:
            rval,frame = video.read()
            cv2.imwrite('./cache/input_img/' + str(n) + '.jpg',frame)
            n = n + 1
            cv2.waitKey(1)
        except Exception as e:
            break
    video.release()
    print('[*] 视频转换图片成功...')




def photo2video(videoName = './cache/output.mp4',fps = 25,imgs = './cache/output_img'):
    '''将图片转换为视频，图片读取路径./cache/output_img.mp4，视频默认写入路径./movie/output.mp4，待合成图片默认文件夹路径./cache/output_img
    INPUTS
        videoName   视频写入路径
        fps         视频帧率
        imgs        待合成图片文件夹路径
    OUTPUTS
        None
    '''
    #获取图像分辨率
    img = cv2.imread(imgs + '/1.jpg')
    sp = img.shape
    height = sp[0]
    width = sp[1]
    size = (width,height)

    fourcc = VideoWriter_fourcc(*'mp4v')
    videoWriter = VideoWriter(videoName,fourcc,fps,size)
    photos = os.listdir(imgs)
    for photo in range(len(photos)):
        frame = cv2.imread(imgs + '/' + str(photo) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    #合成新生成的视频和音频
    if (os.path.exists('./movie/output.mkv')):
        os.remove('./movie/output.mkv')
    command = 'ffmpeg -i ./movie/output.mp4 -i ./cache/audio/audio.wav -c copy ./movie/output.mkv'
    subprocess.call(command,shell=True,stdout= open('/dev/null','w'),stderr=subprocess.STDOUT)
    #os.remove('./cache/output.mp4')
    print('[*] 生成视频成功，路径为./cache/output.mkv')




'''test'''
if __name__ == '__main__':
    photo2video(imgs='./cache/output_img')