---
layout: post
title:  "学习之路：MSELoss + torch广播机制的小陷阱"
date:   2025-12-07 11:38:00 +0800
categories: torch broadcasting
---

周五晚上回家，花了半个小时实现线性回归模型，并且下载boston housing price的数据集。昨天晚上把每一列数据都跟房价做一次线性回归，发现诡异的事情发生了：
```math
 medv = (10e-5) * x + 22.53
```
并且跑10w次SGD后的loss都是72左右。

换言之：所有的自变量，都跟房价呈不相关。（另外可以思考截距为什么是22.53）

# 上代码

```python
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class LinearRegressionModule(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.register_parameter('weights', nn.Parameter(torch.rand((num_features, 1)) * 0.01))
        self.register_parameter('bias', nn.Parameter(torch.randn(1) * 0.01))
    
    def forward(self, inputs):
        return torch.matmul(inputs, self.weights) + self.bias
    
def train(inputs, targets, total_epoches=100000):
    mod = LinearRegressionModule(inputs.shape[1])
    optimizer = optim.AdamW(mod.parameters(), lr=0.001)

    for step in range(0, total_epoches):
        outputs = mod.forward(inputs)
        loss = F.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % 10 == 0):
            print(f"step={step} loss={loss}")
            if loss < 0.001:
                print(f"stopped at step {step}")
                if step % 10 != 0:
                    print(f"step={step} loss={loss}")
                break
    print(f"results: y = x_i * {mod.weights} + {mod.bias.item()}")
    return mod.weights, mod.bias

def main():
    df = pd.read_csv('./Boston.csv', skiprows=0)
    data = torch.tensor(df.values, dtype=torch.float32)
    inputs = data[:,13:14] # lstat
    medv = data[:,14]
    train(inputs, medv)

if __name__ == "__main__":
    main()
```
代码看起来也很简单，百思不得其解，有几个怀疑：
1. 单变量跟房价没有强的线性关系？
2. 把所有自变量跟房价做多元线性回归，发现还是一样的结果
3. 实现错了

今天早上忍不住去看了一下别人的分析，确定lstat跟medv是相关的，系数应该是0.几，差了1万倍，那只能是实现错了。

# 瞪眼法找问题
瞪眼看代码，脑子是解析器。瞪了半个小时，突然发现，为什么medv那个取数的地方，inputs是data[:, 13:14], 而medv是data[:,14]？这两个写法有点不一样，为什么不一样？
最近对矩阵的形状比较敏感，问题是不是在这里？

## 隔离
为了避免每次都随机生成weights/bias对分析的影响，改代码，先生成一个固定weights/bias，保存起来在文件中，后面都加载这个文件以固定weights/bias。

并且在计算完mse_loss()后加了一句打印：
```python
print(f"inputs.shape={inputs.shape}, outputs.shape={outputs.shape}, targets.shape={targets.shape}, loss={loss}")
return
```

## 发现问题
当medv = data[:, 14], 输出的是：
```
inputs.shape=torch.Size([506, 1]), outputs.shape=torch.Size([506, 1]), targets.shape=torch.Size([506]), loss=496.77008056640625
```

当medv = data[:, 14:15], 输出的是：
```
inputs.shape=torch.Size([506, 1]), outputs.shape=torch.Size([506, 1]), targets.shape=torch.Size([506, 1]), loss=497.21795654296875
```

在数据完全一致的情况下，两种方式算出来的mse loss不一样！问题来了。唯一的区别是一个targets（即medv）的形状, 一个targets.shape是[506], 另外一个targets.shape是[506, 1]！

## 复现
写个简单的case来复现这个问题
```python
def reproduce_problem():
    a = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
    ])
    b = torch.tensor([4.0, 5.0, 6.0])
    bt = torch.tensor([
        [4.0],
        [5.0],
        [6.0],
    ])
    print(f"a.shape={a.shape}, b.shape={b.shape}, bt.shape={bt.shape}")
    print(f"a_b_mseloss={F.mse_loss(a, b)}, a_bt_mseloss={F.mse_loss(a, bt)}")
```
运行一下复现了上面的问题：
```
a.shape=torch.Size([3, 1]), b.shape=torch.Size([3]), bt.shape=torch.Size([3, 1])
a_b_mseloss=10.333333015441895, a_bt_mseloss=9.0
```
可以看到a-bt的结果是对的，而a-b的结果是错误的。

# 原因：广播机制的小陷阱
mse的计算过程是: 
```math
mse = \sigma( (a - b)^2 ) / N
```
我们来看看两张情况下a-b分别是什么：
```python
a = [[1],
     [2],
     [3]]          # shape [3, 1]

b = [4, 5, 6]      # shape [3]

bt = [[4],
      [5],
      [6]]         # shape [3, 1]
```

**a-b触发了广播规则**

Step 1: Broadcasting 行为

a: shape [3, 1] → 列向量

b: shape [3] → 行向量（在广播中被视为 [1, 3]）


根据 NumPy/PyTorch 广播规则：

从右对齐维度：

a: dims = (3, 1)

b: dims = (3,) → 自动补齐为 (1, 3)

每个维度取 max：

dim0: max(3, 1) = 3

dim1: max(1, 3) = 3

结果 shape: (3, 3)

所以：

a 被复制成 3 列（每行重复）
b 被复制成 3 行（每列重复）
即：

```python
a_broadcasted = [
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
]

b_broadcasted = [
    [4, 5, 6], 
    [4, 5, 6], 
    [4, 5, 6], 
]

a - b = a_broadcasted - b_broadcasted = [
    [-3, -4, -5],
    [-2, -3, -4],
    [-1, -2, -3],
]

mse(a - b) = (9 + 16 + 25 + 4 + 9 + 16 + 1 + 4 + 9) / 9 = 10.333333333333334
```

PyTorch 不知道我们想要‘逐样本对应’，它只按张量维度规则广播，因此形状必须显式对齐。

真相大白！继续努力:-)



