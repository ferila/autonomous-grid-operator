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