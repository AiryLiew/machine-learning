import numpy

for i in range(3):
    a=numpy.random.RandomState(0)#随机数种子
    print(a.randn(1,10))
    # print(numpy.random.randn(1,10))