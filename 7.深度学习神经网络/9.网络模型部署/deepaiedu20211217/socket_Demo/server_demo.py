import socket  # 导入 socket 模块
import struct
s = socket.socket()  # 创建 socket 通信对象
host = socket.gethostname()  # 获取本地服务器主机名
port = 8000  # 设置端口（网关地址：门），一个主机可以有很多个端口
s.bind((host, port))  # 让通信对象绑定主机和端口地址

s.listen(5)  # 等待客户端连接，每隔5秒监听一次

# byte=b'hello world!'
# byte=bytes('hello world!',encoding='utf-8')
byte=struct.pack("i",2)
while True:
    # 建立客户端连接，等待客户端访问，直到有客户端访问的时候，拿到访问者的地址
    c, addr = s.accept()  #
    print('连接地址：', addr)#客户端的地址
    c.send(byte)#向客户端发送信息
    c.close()  # 关闭连接