import torch
x=torch.arange(12)
X=x.reshape(3,4)
print(x.numel())

# yield
# yield返回一个可迭代的 generator（生成器）对象，可以使用for循环或者调用next()方法遍历生成器对象来提取结果。

def generator(num):
    for i in range(num):
        yield i

res=generator(10) # <generator object generator at 0x000002C0D33CCE40>

print(next(res))