#导入WISG(Web Server Gateway Interface)
from wsgiref.simple_server import make_server
import urllib.parse as urlparse

#application()函数是Python中符合WSGI标准的一个HTTP处理函数,返回是一个字符串
def application(environ,start_response):
     #start_response如下调用就会发送HTTP响应的Header，注意只能调用一次start_response()函数发送Header。
     #start_response()函数两个参数，一是HTTP响应码，一是一组list表示的HTTP Header，每个Header用一个包含两个str的数组表示
     status='200 OK'
     response_headers = [('Content-type', 'text/html')]
     start_response(status,response_headers)

     #调用urlparse的parse_qs解析URL参数,并返回字典
     query_args=environ['QUERY_STRING']
     params = urlparse.parse_qs(environ['QUERY_STRING'])

     print(str(params))
     return [str(params).encode('utf-8')]

ip='127.0.0.1'
port=8089
httpd =make_server(ip,port,application)
print("server is started, port is 8089....")
httpd.serve_forever()