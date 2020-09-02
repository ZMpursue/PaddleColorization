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

## 转换图片/视频
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
# PaddleColorization-黑白照片着色

> 将黑白照片着色是不是一件神奇的事情？

> 本项目将带领你一步一步学习将黑白图片甚至黑白影片的彩色化

---

## 开启着色之旅！！！

### 先来看看成品

![](https://ai-studio-static-online.cdn.bcebos.com/bae1ccceaa4b4c34af7a3f1667547380caabb5fa54d749ef981150601a587935)

> 欢迎大家fork学习~有任何问题欢迎在评论区留言互相交流哦

> 这里一点小小的宣传，我感兴趣的领域包括迁移学习、生成对抗网络。欢迎交流关注。[来AI Studio互粉吧~等你哦~ ](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/56447)


# 1 项目简介

本项目基于paddlepaddle，结合残差网络（ResNet）,通过监督学习的方式，训练模型将黑白图片转换为彩色图片

---

### 1.1 残差网络（ResNet）

### 1.1.1 背景介绍

ResNet(Residual Network) [15] 是2015年ImageNet图像分类、图像物体定位和图像物体检测比赛的冠军。针对随着网络训练加深导致准确度下降的问题，ResNet提出了残差学习方法来减轻训练深层网络的困难。在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。每个残差模块包含两条路径，其中一条路径是输入特征的直连通路，另一条路径对该特征做两到三次卷积操作得到该特征的残差，最后再将两条路径上的特征相加。

残差模块如图9所示，左边是基本模块连接方式，由两个输出通道数相同的3x3卷积组成。右边是瓶颈模块(Bottleneck)连接方式，之所以称为瓶颈，是因为上面的1x1卷积用来降维(图示例即256->64)，下面的1x1卷积用来升维(图示例即64->256)，这样中间3x3卷积的输入和输出通道数都较小(图示例即64->64)。

![](https://ai-studio-static-online.cdn.bcebos.com/7ede3132804549228b5c4a729d90e6b25821272dd9e74e41a95d3363f9e06c0e)

### 1.2 项目设计思路及主要解决问题

> * 设计思路：通过训练网络对大量样本的学习得到经验分布（例如天空永远是蓝色的，草永远是绿色的），通过经验分布推得黑白图像上各部分合理的颜色

> * 主要解决问题：大量物体颜色并不是固定的也就是物体颜色具有多模态性（例如：苹果可以是红色也可以是绿色和黄色）。通常使用均方差作为损失函数会让具有颜色多模态属性的物体趋于寻找一个“平均”的颜色（通常为淡黄色）导致着色后的图片饱和度不高。

---

  ### 1.3 本文主要特征

   * 将Adam优化器beta1参数设置为0.8，具体请参考[原论文](https://arxiv.org/abs/1412.6980)
   * 将BatchNorm批归一化中momentum参数设置为0.5
   * 采用基本模块连接方式
   * 为抑制多模态问题，在均方差的基础上重新设计损失函数

---

 损失函数公式如下:

  $Out = 1/n\sum{(input-label)^{2}} + 26.7/(n{\sum{(input - \bar{input})^{2}}})$

---

 ### 1.4 数据集介绍（ImageNet）

 > ImageNet项目是一个用于视觉对象识别软件研究的大型可视化数据库。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万个图像中，还提供了边界框。ImageNet包含2万多个类别; [2]一个典型的类别，如“气球”或“草莓”，包含数百个图像。第三方图像URL的注释数据库可以直接从ImageNet免费获得;但是，实际的图像不属于ImageNet。自2010年以来，ImageNet项目每年举办一次软件比赛，即ImageNet大规模视觉识别挑战赛（ILSVRC），软件程序竞相正确分类检测物体和场景。 ImageNet挑战使用了一个“修剪”的1000个非重叠类的列表。2012年在解决ImageNet挑战方面取得了巨大的突破，被广泛认为是2010年的深度学习革命的开始。（来源：百度百科）

> ImageNet2012介绍：
>
> * Training images (Task 1 & 2). 138GB.（约120万张高清图片，共1000个类别）
> * Validation images (all tasks). 6.3GB.
> * Training bounding box annotations (Task 1 & 2 only). 20MB.

### 1.5 LAB颜色空间

> Lab模式是根据Commission International Eclairage（CIE）在1931年所制定的一种测定颜色的国际标准建立的。于1976年被改进，并且命名的一种色彩模式。Lab颜色模型弥补了RGB和CMYK两种色彩模式的不足。它是一种设备无关的颜色模型，也是一种基于生理特征的颜色模型。 [1]  Lab颜色模型由三个要素组成，一个要素是亮度（L），a 和b是两个颜色通道。a包括的颜色是从深绿色（低亮度值）到灰色（中亮度值）再到亮粉红色（高亮度值）；b是从亮蓝色（低亮度值）到灰色（中亮度值）再到黄色（高亮度值）。因此，这种颜色混合后将产生具有明亮效果的色彩。（来源：百度百科）
> ![](https://ai-studio-static-online.cdn.bcebos.com/dd030a2a8d22435e815e385d017bf87190ea4cfb1368491f8e08f428104863d0)




# 2.项目总结

通过循序渐进的方式叙述了项目的过程。
对于训练结果虽然本项目通过抑制平均化加大离散程度提高了着色的饱和度，但最终结果仍然有较大现实差距，只能对部分场景有比较好的结果，对人造场景（如超市景观等）仍然表现力不足。
接下来准备进一步去设计损失函数，目的是让网络着色结果足以欺骗人的”直觉感受“，而不是一味地接近真实场景