from flask import Flask,request
import json
import torch
from flask_Demo import module

app = Flask(__name__)#确保Flask调用的是当前模块
net = module.TestModel()
#使用网络
net.eval()
# print(torch.cuda.get_device_name())

#请求的地址为根目录
@app.route("/")
def result():
    x = torch.randn(2,784)
    y = net(x)
    _y = y.detach().numpy()
    return json.dumps(_y.tolist())#将字典、列表转化为字符串


@app.route("/index")
def demo():
    #get请求是通过变量的方式来发送消息的：http://127.0.0.1:5000/index?name=ok
    info=request.args.get("name")
    print(info)
    return "hello world！"

if __name__ == '__main__':
    app.run()