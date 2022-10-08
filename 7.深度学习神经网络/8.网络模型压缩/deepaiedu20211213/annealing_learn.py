import torch
import numpy as np

def Annealing_learning(x,T):
    return np.round(np.exp(x/T)/np.sum(np.exp(x/T)),2)

if __name__ == '__main__':
    data = np.array([13,11,15,18,14,16],dtype=np.float64)
    result1=Annealing_learning(data,100)
    result2=Annealing_learning(data,1)
    result3=Annealing_learning(data,0.1)
    print(result1)
    print(result2)
    print(result3)