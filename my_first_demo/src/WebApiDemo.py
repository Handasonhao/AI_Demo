'''
Author       : Wang.HH
Date         : 2021-06-02 08:19:30
LastEditTime : 2021-06-03 14:48:33
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/src/WebApiDemo.py
'''

class my_app:
  def __init__(self,environ,start_response):
    self.environ = environ
    self.start = start_response
  
  def __iter__(self):
    path = self.environ["PATH_INFO"]
    if path == '/':
      return iter(self.Get_Index())
    elif path == '/hello':
      return iter(self.Get_Hello())
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

  def NotFound(self):
    status = '404 Not Found'
    response_headers = [('Content-type', 'text/plain')]
    self.start(status,response_headers)
    return ["Not Found!\n".encode('utf-8')]