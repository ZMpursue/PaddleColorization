import numpy as np
import os
import skimage.color as color
import matplotlib.pyplot as plt
import conversion
import gray2color
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='基于paddlepaddle的黑白影片色彩重建')
    parser.add_argument('-img_in',dest='img_in',help='输入需要转换的黑白照片路径', type=str)
    parser.add_argument('-save',dest='save',help='需要保存结果的路径', type=str, default='./movie/')
    parser.add_argument('-movie_in',dest='movie_in',help='输入需要转换的黑白影片的路径', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.img_in is not None and args.movie_in is None:
        gray2color.run(input=args.img_in,output=args.save)
        print('[*] 图片转换成功，路径为' + args.save)
    elif args.img_in is None and args.movie_in is not None:
        print('[*] 正在转换中...')
        conversion.video2photo(videoName=args.movie_in)
        cacheIN = './cache/input_img'
        cacheOUT = './cache/ouput_img'
        imgs =  os.listdir(cacheIN)
        for img in imgs:
            gray2color.run(input=cacheIN + img,output= cacheOUT + img)
        conversion.photo2video(videoName=args.save,fps=25)
        print('[*] 视频转换成功，路径为' + args.save)
    else:
        print('[*] 输入有误')


