from flask import Flask
"""
在python中以@开头的函数称为装饰器，装饰器就是接收函数的函数,
装饰器和普通函数不一样，普通函数接收的是变量，返回的是一个或多个值，
而装饰器接收的是一个函数，返回的也是一个函数，而非函数的结果。
"""
"""
普通装饰器不能接收任何参数，但是@app.route()是需要接收参数的
在代码运行期间，动态的增加函数的功能的方式，称为“装饰器”。
"""
app = Flask(__name__)#确保Flask调用的是当前模块
# @app.route("/")
# @app.route("/",methods=["GET"])
#服务端发送默认是get请求，可以改成post请求，但是客户端页面只支持get请求
# @app.route("/",methods=["POST"])#页面默认不支持post请求
#get请求：优点是较快，缺点是长度有限，数据透明不安全
#post请求：优点是传输数据较大，缺点是要以表单的形式发送
@app.route("/",methods=["GET","POST"])#可以同时输入两种方式
def demo():
    return "hello world！"

if __name__ == '__main__':
    app.run()