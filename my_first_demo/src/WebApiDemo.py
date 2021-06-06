'''
Author       : Wang.HH
Date         : 2021-06-02 08:19:30
LastEditTime : 2021-06-06 11:07:35
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/src/WebApiDemo.py
'''
# -*- coding: utf-8 -*-
import urllib.parse as urlparset
import json
import cgi,cgitb # https://www.runoob.com/python3/python3-cgi-programming.html
import ast #使用 ast.literal_eval 进行转换既不存在使用 json 进行转换的问题，也不存在使用 eval 进行转换的 安全性问题，因此推荐使用 ast.literal_eval

class my_app:
  def __init__(self,environ,start_response):
    
    self.environ = environ
    self.start = start_response
  
  def __iter__(self):
    path = self.environ["PATH_INFO"]
    print('path',path)
    if path == '/':
      return iter(self.Get_Index())
      #为什么这里需要再套一层iter朔源，具体解析可以查看http://www.cocoachina.com/articles/88778
    elif path == '/hello':
      return iter(self.Get_Hello())
    elif path == '/json':
      return iter(self.Get_Json())
    elif path == '/back':
      return iter(self.Get_Back())
    elif path == '/login':
      return iter(self.Get_Login())
    elif path == '/post':
      return iter(self.Post())
    else:
      return iter(self.NotFound())
  
  def Get_Index(self):
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    self.start(status,response_headers)
    return ["hello shit!\n".encode('utf-8')]
  
  def Get_Hello(self):
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    self.start(status,response_headers)
    return ["hello world!\n".encode('utf-8')]

  def Get_Json(self):
    status = '200 OK'
    response_headers = [('Content-type', 'text/json')]
    self.start(status,response_headers)
    params = [{'key':12,'value':'iiiii','childre':{'key':12,'value':'iiiii'}}]
    print(json.dumps(params))
    return [json.dumps(params).encode('utf-8')]
  
  def Get_Back(self):
    status = '200 OK'
    response_headers = [('Content-type', 'text/plain')]
    self.start(status,response_headers)
    #调用urlparset的parse_qs解析URL参数,并返回字典
    query_args=self.environ['QUERY_STRING']
    params = urlparset.parse_qs(self.environ['QUERY_STRING'])
    print(str(params))
    return [str(params).encode('utf-8')]
  
  def Get_Login(self):
    status = '200 OK'
    response_headers = [('Content-type', 'text/html')]
    self.start(status,response_headers)
    #调用urlparset的parse_qs解析URL参数,并返回字典
    query_args=self.environ['QUERY_STRING']
    params = urlparset.parse_qs(query_args)
    name = params['name'][0]
    pwd = params['pwd'][0]
    tel = params['tel'][0]
    print(params)
    print(name)
    if name=='admin' and pwd=='123456' and tel=='1234567890':
      result = {'code':200,'msg':'you get the flag!'}
      return [json.dumps(result).encode('utf-8')]
    else:
      result = {'code':404,'msg':'worning!'}
      return [json.dumps(result).encode('utf-8')]
    
  def Post(self):
    status = '200 OK'
    response_headers = [('Content-type', 'application/json')]
    self.start(status,response_headers)
    try:
      request_body_size = int(self.environ['CONTENT_LENGTH'])
    except:
      request_body_size = 0
    print(request_body_size)
    request_body = self.environ['wsgi.input'].read(request_body_size).decode()
    request_body = ast.literal_eval(request_body)
    print(request_body)
    name = request_body['name']
    pwd = request_body['pwd']
    print(name,pwd)
    return [json.dumps(request_body).encode('utf-8')]

  def NotFound(self):
    status = '404 Not Found'
    response_headers = [('Content-type', 'text/plain')]
    self.start(status,response_headers)
    return ["Not Found!\n".encode('utf-8')]