'''
Author       : Wang.HH
Date         : 2021-04-28 10:39:59
LastEditTime : 2021-05-27 17:01:54
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/first.py
'''
print('<--加载工具库开始-->')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import skimage.transform as tf
import os
print('<--加载工具库完成-->')

file_name = os.path.dirname(__file__)
#获取项目根目录路径

def load_dataset():
  f = h5py.File(file_name+"/datasets/hhhh.h5","w") 
  # h5py在windows下读写文件需要采用绝对路径；Mac中不要简单地从文件系统复制文件路径；请尝试手动输入，linux不存在上述问题
  train_dataset = h5py.File(file_name+'/datasets/train_catvnoncat.h5','r')#加载训练数据
  print("train_set_x:{}".format(train_dataset["train_set_x"][:]))
  train_set_x_orig = np.array(train_dataset["train_set_x"][:])#取训练特征数据集
  train_set_y_orig = np.array(train_dataset["train_set_y"][:])#取训练标签数据集
  test_dataset = h5py.File(file_name+'/datasets/test_catvnoncat.h5','r')#加载测试数据
  test_set_x_orig = np.array(test_dataset["test_set_x"][:])#取测试特征数据集
  test_set_y_orig = np.array(test_dataset["test_set_y"][:])#取测试标签数据集
load_dataset()
