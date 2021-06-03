'''
Author       : Wang.HH
Date         : 2021-06-02 11:08:37
LastEditTime : 2021-06-02 12:55:03
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/src/webserve.py
'''
import wsgiref.simple_server as wss
from WebApiDemo import simple_app as app

mk = wss.make_server
demo_app = wss.demo_app
httpd = mk('127.0.0.1',8086,demo_app)
sa = httpd.socket.getsockname()
print(httpd.socket)
print(sa)
print('http://{0}:{1}/'.format(*sa))
httpd.serve_forever()