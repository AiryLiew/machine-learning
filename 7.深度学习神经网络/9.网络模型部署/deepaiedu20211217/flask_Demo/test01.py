# This is our decorator

def simple_decorator(f):
    # This is the new function we're going to return
    # This function will be used in place of our original definition
    def wrapper():
        print("Entering Function")
        f()
        print("Exited Function")
    #返回的是一个函数
    return wrapper

#一般的装饰器不能接收参数
@simple_decorator
def hello():#f()
    print("Hello World")

hello()