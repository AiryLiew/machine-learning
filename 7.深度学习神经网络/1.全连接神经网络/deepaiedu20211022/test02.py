import torch
import numpy as np

def one_hot(h,w,arr):
    zero_arr=np.zeros([h,w])
    #循环每一行
    for i in range(h):
        #具体的值
        j=arr[i]
        #找到每一行的具体的值的索引，然后赋值为1
        zero_arr[i][j]=1
    return zero_arr

if __name__ == '__main__':
    arr=np.array([5,2,8,6])
    # output=one_hot(len(arr),max(arr)+1,arr)
    # print(output)
    # print(np.argmax(output,1))
    zero_arr=torch.zeros(len(arr),max(arr)+1)
    zero_arr[torch.arange(len(arr)),arr]=1
    # print(torch.arange(5))
    # torch.zeros(len(arr), max(arr) + 1)[torch.arange(len(arr)),arr]=1
    print(zero_arr)
