from flask import Flask,request
from PIL import Image
import io
app = Flask(__name__)#确保Flask调用的是当前模块

@app.route("/",methods=["post"])
def demo():
    #服务端获取客户端发送的图片，并且展示出来
    info=request.form.get("name")
    file = request.files.get("filename")
    img_byte = file.read()
    image = Image.open(io.BytesIO(img_byte))
    print("接收到客户端的上传的图片 "+info)
    image.show()
    return info+" 图片已上传成功！"

if __name__ == '__main__':
    app.run()


