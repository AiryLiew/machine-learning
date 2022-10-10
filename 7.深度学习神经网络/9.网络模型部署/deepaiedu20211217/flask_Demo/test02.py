def decorator_factory(enter_message, exit_message):
    # We're going to return this decorator
    def simple_decorator(f):
        def wrapper():
            print(enter_message)
            f()
            print(exit_message)
        return wrapper
    return simple_decorator

#让装饰器可以接收参数，其实就是装饰器的装饰器
@decorator_factory("Start", "End")
def hello():#f()
    print("Hello World")

hello()