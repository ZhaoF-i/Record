# [Paper Summary](https://github.com/FeiF-i/Record)

| 题目 | 分类 | 时间 |
| :--- | ---- | ---- |

* 动机
* 方法

# Audio, Speech, and Language Processing
## 传统方法
### 单通道
- 谱减法

- 维纳滤波

- MMSE

- 卡尔曼滤波
### 多通道
- 

##  Speech Enhancement

-  A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement，网络结构，2022/1/14
	-  动机：实现低延时或零延时的语音增强应用于现实应用中，如助听器、可植入耳蜗
	-  方法：提出了因果卷积encoder、decoder和LSTM组成的CRN结构
	
	
- A Deep Learning-based Kalman Filter for Speech Enhancement，传统方法+神经网络，2022/1/14
	- 动机：现有的卡尔曼滤波器(KF)在真实噪声条件下对噪声方差和线性预测系数(LPCs)的估计较差。导致增强效果差
	- 方法：采用了一种基于MMSE的噪声功率谱密度(PSD)估计的深度学习方法，称为DeepMMSE。利用估计的噪声PSD来计算噪声的方差；构造了一个白化滤波器，其系数由估计的噪声PSD计算出来。然后将其应用于有噪声的语音中，生成用于计算lpc的预白化语音。
	  ![](D:\tools\typora\picture\image-20220114160310054.png)

-  A Maximum Likelihood Approach to SNR-Progressive Learning Using Generalized Gaussian Distribution for LSTM-Based Speech Enhancement，loss，2022/1/14
	- 动机：
	- 方法：

## Speech Separation

- 题目
	- 动机：
	
	* 方法：
- 题目
	- 动机
	- 方法

## ASR

- 题目
	- 动机：
	- 方法：
- 题目
	- 动机
	- 方法
## VAD

- 题目
	- 动机：
	
	* 方法：
- 题目
	- 动机
	- 方法


# CV
- 题目
	- 动机：
	
	- 方法：
- 题目
	- 动机
	- 方法
# NLP
- 题目
	- 动机：
	- 方法：
- 题目
	- 动机
	- 方法