#客户端
import socket
s = socket.socket()
host = socket.gethostname()
port = 8000
s.connect((host, port)) #客户端接入
while True:
    mes = str(s.recv(1024),encoding = "utf-8")
    print("服务器：{}".format(mes))
    if mes == "再见":
        break
    mes = input("客户端：")
    s.send(bytes(mes,encoding="utf-8"))
s.close()

"""
若是你使用上面的代码，在不同的计算机上运行，必然是会报错的，
那是因为，我们的服务器和客户端的host都是socket.gethostname()，
这句代码是用来获取本机的hostname的，
很显然，如果你想实现两台设备之间的通讯，则必须把客户端的host改为服务器端的host。
如果某一方连续发送信息，改如何解决？
"""