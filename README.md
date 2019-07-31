![image](resouse/logo.png)

**通过黑白图片的情景语义找到颜色和结构纹理特征的映射,将黑白影片彩色化**

| 脚本需要配合命令行工具ffmpeg |
| -------------------------- |
## 项目文件结构

+ cache 视频转换缓存目录
   * audio 原视频的音频文件
   * input_img 原视频分解成每帧图像目录(黑白图片)
   * output_img 将input_img文件夹下图片处理后生成对应的图片目录(彩色图片)
+ data 训练集目录
	* test 测试集
	* train 训练集
+ model 模型存放目录
+ movie 视频文件(*.mp4)文件存放目录
+ prior 预处理目录
    * Clear.py 清理训练集和测试集中的损坏文件和不是JPEG格式的图片
    * getExperience.py 预处理ImageNet数据集得到Q空间上的经验分布
    * getWeight.py 将经验分布结合高斯函数，经过处理得到经验分布的权值
    * imageCrop.py 将图片进行缩放后再裁剪到256*256像素
    * Experience.npy 运行getExperience.py生成经验分布的Numpy矩阵文件
    * Q.npy 人为划定的Q空间对应的Numpy矩阵文件
    * Weight.npy 运行getWeight.py生成经验分布的Numpy矩阵文件
+ train 训练脚本目录
    * fileReader.npy 从指定文件夹读取文件的Reader，并做预处理
    * train.npy 训练脚本
+ conversion.py 将movie目录读取input.mp4文件并分解成每帧图片存放在cache/input_img/,音频文件存放在cache/audio/
+ gray2color.py 将输入图片经过模型处理再转换为RGB图片并储存

##转换图片/视频
先进入paddle环境
```bash
cd <项目路径>
source activate <ENV>
```
转换图片
```bash
Python main.py -img_in <需要转换图片路径> -save <保存位置>
```
转换视频
```bash
Python main.py -movie_in <需要转换视频路径> -save <保存位置>
```
