from flask import Flask,request

app = Flask(__name__)#确保Flask调用的是当前模块

@app.route("/",methods=["post"])
def demo():
    info=request.form.get("name")
    print("收到客户端的信息："+info)
    return info+" 已提交至服务器！"

if __name__ == '__main__':
    app.run()


