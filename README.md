# google云端硬盘与colab连接，在colab端训练GoogLeNet。（基础入门版模型搭建、模型训练和模型验证。实现pytorch架构下自带的单通道灰度图“FashionMINIST”中的衣物分类。训练10次，准确率达到93.7。  ）
## 首先将代码上传至google云端硬盘；  
### 使用colab与云端硬盘建立连接，实现代码如下：   

```
@Override
from google.colab import drive

# 挂载Google云端硬盘
# This error often occurs due to authentication issues.
# Potential causes: revoked access, network problems, or multiple Google accounts logged in.
# To fix: Check your Google account, revoke and re-authorize access in Google Account settings, or restart the runtime.
drive.mount('/content/drive')

# 验证挂载是否成功
!ls /content/drive/My\ Drive
```  
当提示：
<img width="1099" height="320" alt="image" src="https://github.com/user-attachments/assets/9b80f124-14db-4359-b684-e3d931b96ef0" />
与云端硬盘连接成功。

### 随后查看目前所在位置

```
@Override
!pwd
```
运行后看到:
<img width="830" height="84" alt="image" src="https://github.com/user-attachments/assets/4aa1b1b0-94e1-43c3-82a2-91171f563e41" />

### 访问drive文件夹
<img width="355" height="99" alt="image" src="https://github.com/user-attachments/assets/8904854d-692b-485b-95f5-6f46ae9097c8" />

### 查看当前文件夹下的文件
<img width="786" height="85" alt="image" src="https://github.com/user-attachments/assets/c5fa8a2b-05df-4755-83cc-cceebd7c0d35" />

### 访问Mydrive文件夹
<img width="370" height="94" alt="image" src="https://github.com/user-attachments/assets/f983547b-6c88-4a37-a898-e76ee0860a50" />

### 访问GoogLeNet文件夹
<img width="913" height="105" alt="image" src="https://github.com/user-attachments/assets/6584a1f0-9596-4fdc-a882-fba42412cf8e" />

### 运行model.py文件夹

```
@Override
!python model.py
```
得到结果，查看到GoogLeNet的整个前向传播的过程，以及网络参数量  
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           3,200
              ReLU-2         [-1, 64, 112, 112]               0
         MaxPool2d-3           [-1, 64, 56, 56]               0
            Conv2d-4           [-1, 64, 56, 56]           4,160
              ReLU-5           [-1, 64, 56, 56]               0
            Conv2d-6          [-1, 192, 56, 56]         110,784
              ReLU-7          [-1, 192, 56, 56]               0
         MaxPool2d-8          [-1, 192, 28, 28]               0
            Conv2d-9           [-1, 64, 28, 28]          12,352
             ReLU-10           [-1, 64, 28, 28]               0
           Conv2d-11           [-1, 96, 28, 28]          18,528
             ReLU-12           [-1, 96, 28, 28]               0
           Conv2d-13          [-1, 128, 28, 28]         110,720
             ReLU-14          [-1, 128, 28, 28]               0
           Conv2d-15           [-1, 16, 28, 28]           3,088
             ReLU-16           [-1, 16, 28, 28]               0
           Conv2d-17           [-1, 32, 28, 28]          12,832
             ReLU-18           [-1, 32, 28, 28]               0
        MaxPool2d-19          [-1, 192, 28, 28]               0
           Conv2d-20           [-1, 32, 28, 28]           6,176
             ReLU-21           [-1, 32, 28, 28]               0
        Inception-22          [-1, 256, 28, 28]               0
           Conv2d-23          [-1, 128, 28, 28]          32,896
             ReLU-24          [-1, 128, 28, 28]               0
           Conv2d-25          [-1, 256, 28, 28]          65,792
             ReLU-26          [-1, 256, 28, 28]               0
           Conv2d-27          [-1, 192, 28, 28]         442,560
             ReLU-28          [-1, 192, 28, 28]               0
           Conv2d-29           [-1, 32, 28, 28]           8,224
             ReLU-30           [-1, 32, 28, 28]               0
           Conv2d-31           [-1, 96, 28, 28]          76,896
             ReLU-32           [-1, 96, 28, 28]               0
        MaxPool2d-33          [-1, 256, 28, 28]               0
           Conv2d-34           [-1, 64, 28, 28]          16,448
             ReLU-35           [-1, 64, 28, 28]               0
        Inception-36          [-1, 480, 28, 28]               0
        MaxPool2d-37          [-1, 480, 14, 14]               0
           Conv2d-38          [-1, 192, 14, 14]          92,352
             ReLU-39          [-1, 192, 14, 14]               0
           Conv2d-40           [-1, 96, 14, 14]          46,176
             ReLU-41           [-1, 96, 14, 14]               0
           Conv2d-42          [-1, 208, 14, 14]         179,920
             ReLU-43          [-1, 208, 14, 14]               0
           Conv2d-44           [-1, 16, 14, 14]           7,696
             ReLU-45           [-1, 16, 14, 14]               0
           Conv2d-46           [-1, 48, 14, 14]          19,248
             ReLU-47           [-1, 48, 14, 14]               0
        MaxPool2d-48          [-1, 480, 14, 14]               0
           Conv2d-49           [-1, 64, 14, 14]          30,784
             ReLU-50           [-1, 64, 14, 14]               0
        Inception-51          [-1, 512, 14, 14]               0
           Conv2d-52          [-1, 160, 14, 14]          82,080
             ReLU-53          [-1, 160, 14, 14]               0
           Conv2d-54          [-1, 112, 14, 14]          57,456
             ReLU-55          [-1, 112, 14, 14]               0
           Conv2d-56          [-1, 224, 14, 14]         226,016
             ReLU-57          [-1, 224, 14, 14]               0
           Conv2d-58           [-1, 24, 14, 14]          12,312
             ReLU-59           [-1, 24, 14, 14]               0
           Conv2d-60           [-1, 64, 14, 14]          38,464
             ReLU-61           [-1, 64, 14, 14]               0
        MaxPool2d-62          [-1, 512, 14, 14]               0
           Conv2d-63           [-1, 64, 14, 14]          32,832
             ReLU-64           [-1, 64, 14, 14]               0
        Inception-65          [-1, 512, 14, 14]               0
           Conv2d-66          [-1, 128, 14, 14]          65,664
             ReLU-67          [-1, 128, 14, 14]               0
           Conv2d-68          [-1, 128, 14, 14]          65,664
             ReLU-69          [-1, 128, 14, 14]               0
           Conv2d-70          [-1, 256, 14, 14]         295,168
             ReLU-71          [-1, 256, 14, 14]               0
           Conv2d-72           [-1, 24, 14, 14]          12,312
             ReLU-73           [-1, 24, 14, 14]               0
           Conv2d-74           [-1, 64, 14, 14]          38,464
             ReLU-75           [-1, 64, 14, 14]               0
        MaxPool2d-76          [-1, 512, 14, 14]               0
           Conv2d-77           [-1, 64, 14, 14]          32,832
             ReLU-78           [-1, 64, 14, 14]               0
        Inception-79          [-1, 512, 14, 14]               0
           Conv2d-80          [-1, 112, 14, 14]          57,456
             ReLU-81          [-1, 112, 14, 14]               0
           Conv2d-82          [-1, 128, 14, 14]          65,664
             ReLU-83          [-1, 128, 14, 14]               0
           Conv2d-84          [-1, 288, 14, 14]         332,064
             ReLU-85          [-1, 288, 14, 14]               0
           Conv2d-86           [-1, 32, 14, 14]          16,416
             ReLU-87           [-1, 32, 14, 14]               0
           Conv2d-88           [-1, 64, 14, 14]          51,264
             ReLU-89           [-1, 64, 14, 14]               0
        MaxPool2d-90          [-1, 512, 14, 14]               0
           Conv2d-91           [-1, 64, 14, 14]          32,832
             ReLU-92           [-1, 64, 14, 14]               0
        Inception-93          [-1, 528, 14, 14]               0
           Conv2d-94          [-1, 256, 14, 14]         135,424
             ReLU-95          [-1, 256, 14, 14]               0
           Conv2d-96          [-1, 160, 14, 14]          84,640
             ReLU-97          [-1, 160, 14, 14]               0
           Conv2d-98          [-1, 320, 14, 14]         461,120
             ReLU-99          [-1, 320, 14, 14]               0
          Conv2d-100           [-1, 32, 14, 14]          16,928
            ReLU-101           [-1, 32, 14, 14]               0
          Conv2d-102          [-1, 128, 14, 14]         102,528
            ReLU-103          [-1, 128, 14, 14]               0
       MaxPool2d-104          [-1, 528, 14, 14]               0
          Conv2d-105          [-1, 128, 14, 14]          67,712
            ReLU-106          [-1, 128, 14, 14]               0
       Inception-107          [-1, 832, 14, 14]               0
       MaxPool2d-108            [-1, 832, 7, 7]               0
          Conv2d-109            [-1, 256, 7, 7]         213,248
            ReLU-110            [-1, 256, 7, 7]               0
          Conv2d-111            [-1, 160, 7, 7]         133,280
            ReLU-112            [-1, 160, 7, 7]               0
          Conv2d-113            [-1, 320, 7, 7]         461,120
            ReLU-114            [-1, 320, 7, 7]               0
          Conv2d-115             [-1, 32, 7, 7]          26,656
            ReLU-116             [-1, 32, 7, 7]               0
          Conv2d-117            [-1, 128, 7, 7]         102,528
            ReLU-118            [-1, 128, 7, 7]               0
       MaxPool2d-119            [-1, 832, 7, 7]               0
          Conv2d-120            [-1, 128, 7, 7]         106,624
            ReLU-121            [-1, 128, 7, 7]               0
       Inception-122            [-1, 832, 7, 7]               0
          Conv2d-123            [-1, 384, 7, 7]         319,872
            ReLU-124            [-1, 384, 7, 7]               0
          Conv2d-125            [-1, 192, 7, 7]         159,936
            ReLU-126            [-1, 192, 7, 7]               0
          Conv2d-127            [-1, 384, 7, 7]         663,936
            ReLU-128            [-1, 384, 7, 7]               0
          Conv2d-129             [-1, 48, 7, 7]          39,984
            ReLU-130             [-1, 48, 7, 7]               0
          Conv2d-131            [-1, 128, 7, 7]         153,728
            ReLU-132            [-1, 128, 7, 7]               0
       MaxPool2d-133            [-1, 832, 7, 7]               0
          Conv2d-134            [-1, 128, 7, 7]         106,624
            ReLU-135            [-1, 128, 7, 7]               0
       Inception-136           [-1, 1024, 7, 7]               0
AdaptiveAvgPool2d-137           [-1, 1024, 1, 1]               0
         Flatten-138                 [-1, 1024]               0
          Linear-139                   [-1, 10]          10,250
================================================================
Total params: 6,181,930
Trainable params: 6,181,930
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 70.97
Params size (MB): 23.58
Estimated Total Size (MB): 94.74
-------------------------------------------------------------
```
### 运行model_train.py进行模型训练，在训练10轮次后准确率达到93.77%  
```
Epoch 0/9
__________
0 train loss: 0.6698 train Acc: 0.7477
0 val loss: 0.3918 val Acc: 0.8536
训练和验证集的耗费时间 3 m 21 s
Epoch 1/9
__________
1 train loss: 0.3677 train Acc: 0.8639
1 val loss: 0.3126 val Acc: 0.8807
训练和验证集的耗费时间 6 m 46 s
Epoch 2/9
__________
2 train loss: 0.3082 train Acc: 0.8833
2 val loss: 0.2881 val Acc: 0.8954
训练和验证集的耗费时间 10 m 11 s
Epoch 3/9
__________
3 train loss: 0.2725 train Acc: 0.8982
3 val loss: 0.2637 val Acc: 0.9018
训练和验证集的耗费时间 13 m 36 s
Epoch 4/9
__________
4 train loss: 0.2429 train Acc: 0.9099
4 val loss: 0.2384 val Acc: 0.9127
训练和验证集的耗费时间 16 m 60 s
Epoch 5/9
__________
5 train loss: 0.2233 train Acc: 0.9156
5 val loss: 0.2444 val Acc: 0.9133
训练和验证集的耗费时间 20 m 23 s
Epoch 6/9
__________
6 train loss: 0.2012 train Acc: 0.9255
6 val loss: 0.2442 val Acc: 0.9117
训练和验证集的耗费时间 23 m 46 s
Epoch 7/9
__________
7 train loss: 0.1895 train Acc: 0.9292
7 val loss: 0.2474 val Acc: 0.9097
训练和验证集的耗费时间 27 m 9 s
Epoch 8/9
__________
8 train loss: 0.1786 train Acc: 0.9337
8 val loss: 0.2462 val Acc: 0.9128
训练和验证集的耗费时间 30 m 32 s
Epoch 9/9
__________
9 train loss: 0.1668 train Acc: 0.9377
9 val loss: 0.2323 val Acc: 0.9163
训练和验证集的耗费时间 33 m 55 s
Figure(1200x400)
```
### 最后运行model_test.py进行模型测试。
