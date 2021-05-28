'''
Author       : Wang.HH
Date         : 2021-04-28 10:39:59
LastEditTime : 2021-05-28 16:00:30
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/first.py
'''
print('<--加载工具库开始-->')
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline # 这是在Iphython中使用的magic命令，作用是将画图在前端执行。vscode环境中配置的是标准的Python解释器，所以这语句是不能运行的，Ipython相关介绍见：https://blog.csdn.net/gavin_john/article/details/53086766
# pip install -t d:/anaconda/envs/python38/lib/site-packages pip包名 将pip包安装到指定目录
import h5py
import skimage.transform as tf
import os
print('<--加载工具库完成-->')

file_path = os.path.dirname(__file__)
#获取项目根目录路径

def load_dataset():
  # f = h5py.File(file_path+"/datasets/hhhh.h5","w")
  # h5py在windows下读写文件需要采用绝对路径；Mac中不要简单地从文件系统复制文件路径；请尝试手动输入，linux不存在上述问题
  
  train_dataset = h5py.File(file_path+'/datasets/train_catvnoncat.h5','r')# 加载训练数据
  print("train_set_x:{}".format(train_dataset["train_set_x"]))
  train_set_x_orig = np.array(train_dataset["train_set_x"][:])# 取训练特征数据集
  train_set_y_orig = np.array(train_dataset["train_set_y"][:])# 取训练标签数据集
  
  test_dataset = h5py.File(file_path+'/datasets/test_catvnoncat.h5','r')# 加载测试数据
  test_set_x_orig = np.array(test_dataset["test_set_x"][:])# 取测试特征数据集
  test_set_y_orig = np.array(test_dataset["test_set_y"][:])# 取测试标签数据集
  
  classes = np.array(test_dataset["list_classes"][:])# 加载标签类别数据，此处类别只有两种，1代表有猫，0代表无猫
  print("classes:{}".format(test_dataset["list_classes"][:]))
  
  return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes
def show_rose(values,title):
  #玫瑰花瓣的个数为8，45度
    n = 8
    angle = np.arange(0,2*np.pi,2*np.pi/n)
    #绘制的数据
    radius = np.array(values)
    #极坐标条形图，polar为True
    plt.axes([0,0,1.5,1.5],polar = True)
    
    color = np.random.random(size = 24).reshape((8,3))
    
    plt.bar(angle,radius,color = color)
    
    plt.title(title,loc = 'left')
    
    plt.show()
v = [1,2,3,4,5,6,7,8]
show_rose(v,'test')
load_dataset()


