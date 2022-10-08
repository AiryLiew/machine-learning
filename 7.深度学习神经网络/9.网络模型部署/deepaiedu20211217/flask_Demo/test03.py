class NotFlask():
    def __init__(self):
        self.routes = {}
    def route(self, route_str):
        def decorator(f):
            self.routes[route_str] = f
            return f
        return decorator
    def run(self):
        view_function = self.routes.get("/")
        if view_function:
            return view_function()
        else:
            raise ValueError('Route "{}"" has not been registered'.format("/"))

app = NotFlask()
@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    print(app.run())
