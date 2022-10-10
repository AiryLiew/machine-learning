import socket  # 导入 socket 模块
import struct
s = socket.socket()  # 创建 socket 对象
host = socket.gethostname()  # 获取本地主机名
# host = "192.168.2.5"  # 获取本地主机名
port = 8000  # 设置端口号
buffer_size=1024#每次接收的最大字节数
s.connect((host, port))#连接服务端
data=s.recv(buffer_size)#接收服务端数据
data = struct.unpack("i",data)
print(data)
s.close()