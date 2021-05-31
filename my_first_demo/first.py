'''
Author       : Wang.HH
Date         : 2021-04-28 10:39:59
LastEditTime : 2021-05-31 16:41:52
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/first.py
'''
print('<--加载工具库开始-->')
import numpy as np
import wsgiref.simple_server as ws
print(ws.make_server)
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
  
  train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
  test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
  # print("classes:{}".format(test_dataset["list_classes"][:]))
  
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

# v = [1,2,3,4,5,6,7,8]
# show_rose(v,'test')

load_dataset()
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 30
plt.imshow(train_set_x_orig[index])
plt.show()
print(train_set_y[:, index])
print(classes[np.squeeze(train_set_y[:, index])].decode("utf-8"))


print("train_set_x_orig.shape : " + str(train_set_x_orig.shape))
print("train_set_y.shape : " + str(train_set_y.shape))
print("test_set_x_orig.shape : " + str(test_set_x_orig.shape))
print("test_set_y.shape : " + str(test_set_y.shape))

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]

print("训练样本数：m_train = "+str(m_train))
print("测试样本数：m_test = "+str(m_test))
print("每张图片的宽/高：num_px = "+str(num_px))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T #新数组的shape属性应该要与原来数组的一致，即新数组元素数量与原数组元素数量要相等。一个参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值(实际上就是数组的元素总数除以非-1的参数)。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten.shape : " + str(train_set_x_flatten.shape))
print("test_set_x_flatten.shape : " + str(test_set_x_flatten.shape))

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255
print("数据准备完成！")


def sigmoid(z):
  """
  sigmoid 激活函数

  Args:
      z (array|number): 一个数值或者一个numpy数组

  Returns:
      array|number: 经过sigmoid算法计算后的值，在[0,1]范围内
  """ 
  s = 1 / (1 + np.exp(-z))
  return s

def initialize_with_zeros(dim):
  """
  initialize_with_zeros 用于初始化权重群组w和偏置/阈值b

  Args:
      dim (number): w的大小
      
  Rerurns:
      w (array): 权重数组
      b (number): 偏置bias
  """  
  w = np.zeros((dim,1))
  b = 0
  
  return w, b
def propagate(w, b, X, Y):
  """
  propagate 执行前向传播，计算成本cost；执行反向传播，计算dw和db，为后面的梯度下降做准备

  Args:
      w (nparray): 权重数组，维度(12288, 1) 12288行，1列
      b (nparray): 偏置bias
      X (nparray): 图片特征数据，维度是(12288, 209)
      Y (nparray): 图片对应的标签，0或1，0是无猫，1是有猫，维度是(1,209)

  Returns:
      cost: 成本
      dw： w的梯度
      db: b的梯度
  """  
  m = X.shape[1]
  
  # 前向传播
  A = sigmoid(np.dot(w.T,X) + b)
  cost = -np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A)) / m
  
  # 向后传播
  dZ = A - Y
  dw = np.dot(X, dZ.T) / m
  db = np.sum(dZ) / m
  
  # 保存dw和db到字典里面
  grads = {"dw":dw, "db":db}
  
  return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
  """
  optimize 梯度下降函数，通过梯度下降来更新参数w和b，达到越来越优化的目的

  Args:
      w (nparray): 权重数组，维度（12288，1）
      b (nparray): 偏置bias
      X (nparray)): 图片的特征数据，维度是（12288，1）
      Y (nparray): 图片对应的标签，0或1，0表示无猫，1表示有猫，维度是（1，209）
      num_iterations (number): 优化的次数
      learning_rate (number): 学习步进，控制优化步进的参数
      print_cost (bool, optional): 为true时，每优化100次就把成本cost打印出来，观察成本变化.默认False.

  Returns:
      params: 优化后的w和b
      costs: 每优化100次，将成本记录下来，成本越小，表示参数越优化
  """  
  costs = []
  for i in range(num_iterations):
    grads, cost = propagate(w, b, X ,Y) # 计算得出梯度和成本
    
    # 从字典取出梯度
    dw = grads["dw"]
    db = grads["db"]
    
    # 进行梯度下降，更新参数，使其越来越优化，使成本越来越小
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # 记录成本
    if i % 100 == 0:
      costs.append(cost)
      if print_cost:
        print("优化%i次后成本是：%f"%(i,cost))
    params = {"w":w,"b":b}
  return params, costs

def predict(w, b, X):
  return

# print("test_set_y.shape : " + str(test_set_y.shape))
# print("load_dataset():{}".format(load_dataset()))
