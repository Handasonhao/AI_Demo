'''
Author       : Wang.HH
Date         : 2021-06-02 08:19:30
LastEditTime : 2021-06-03 15:29:05
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/src/WebApiDemo.py
'''
# -*- coding: utf-8 -*-
import urllib.parse as urlparse
import json

class my_app:
  def __init__(self,environ,start_response):
    self.environ = environ
    self.start = start_response
  
  def __iter__(self):
    path = self.environ["PATH_INFO"]
    if path == '/':
      return iter(self.Get_Index())
      #为什么这里需要再套一层iter朔源，具体解析可以查看http://www.cocoachina.com/articles/88778
    elif path == '/hello':
      return iter(self.Get_Hello())
    elif path == '/json':
      return iter(self.Get_Json())
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
    # params = urlparse.parse_qs(environ['QUERY_STRING'])
    params = [{'key':12,'value':'iiiii','childre':{'key':12,'value':'iiiii'}}]
    print(json.dumps(params))
    # return [str(params).encode('utf-8')]
    return [json.dumps(params).encode('utf-8')]

  def NotFound(self):
    status = '404 Not Found'
    response_headers = [('Content-type', 'text/plain')]
    self.start(status,response_headers)
    return ["Not Found!\n".encode('utf-8')]