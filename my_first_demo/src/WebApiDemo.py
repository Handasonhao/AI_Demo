'''
Author       : Wang.HH
Date         : 2021-06-02 08:19:30
LastEditTime : 2021-06-02 12:53:52
LastEditors  : Wang.HH
Description  : your description
FilePath     : /AI_Demo/my_first_demo/src/WebApiDemo.py
'''
def simple_app(envirron, start_response):
  status = '200 OK'
  response_headers = [('Content-type', 'text/plain')]
  start_response(status,response_headers)
  return ["hello shit!\n"]