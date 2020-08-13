import numpy as np
import pandas as pd

class A(object):
    def method1(self):
        return "1"
    
    def method2(self):
        return "method2"

class B(A):
    def method1(self):
        return "2"

if __name__ == "__main__":
    a = A()
    print(a.method1())
    print(a.method2())
    print("---")
    b = B()
    print(b.method1())
    print(b.method2())

    #####3

    n_line = 4
    n_bus = 4
    reactance = 0.8

    Bd = np.identity(n_line)*(1/reactance)

    I = np.zeros((n_bus, n_line))
    I[2,0] = 1
    I[3,0] = -1
    I[0,1] = 1
    I[2,1] = -1
    I[1,2] = 1
    I[2,2] = -1
    I[0,3] = 1
    I[1,3] = -1
    I = I[:-1,:]

    S = np.identity(n_bus)
    S = S[:-1,:-1]

    It = np.transpose(I)
    inverse = np.linalg.inv(I.dot(Bd).dot(It))
    ans = pd.DataFrame(Bd.dot(It).dot(inverse).dot(S))
    print(ans)