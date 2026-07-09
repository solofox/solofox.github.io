---
layout: post
title:  "[模型推理] Inference From Scratch: 使用 torch.nn.Module 来实现模型"
date:   2026-06-25 21:49:00 +0800
categories: llm gpt2 torch-style
---

> 文章含 AI 量：< 10%。AI 主要负责润色。
> 代码含 AI 量：< 5%。

# 使用 torch.nn.Module 来实现模型

前面的版本中，为了展示典型操作的实现方法，我们都选择了完全独立编写模型，不从现有的 torch.nn.Module 中继承，也不使用 torch.nn.Linear 等已有的模块。现在让我们来重构它。

这一期文章比较简单、轻松。

## 从一个算式拆解 torch.nn.Module 的结构

每个使用 pytorch 编写模型的工程师，都应该阅读过这篇入门文章 [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)。当我们照猫画虎地把 MNIST 模型复现完成后，关掉网页，合上电脑，让我们再想想，模型是什么？

这当然有很多种解读方式。`程序 = 数据结构 + 算法`，从工程实现的视角出发，我定义 **模型 = 参数 + 算子 + 结构**。

1. **参数**：参数是数量化的智能，参数也是一系列张量（tensor），是训练时不断被 BP 算法优化的浮点数，是保存在 .safetensors/.ckpt/.pt/.onnx 文件中的浮点数。这些都对，看你在什么场合，面向什么问题讨论它。参数的规模对模型的性能至关重要，Scaling Law。

2. **算子**：算子是典型的操作范式。矩阵乘法，激活函数，正则化，Dropout，Softmax，Loss 都是非常典型的算子。算子的粒度也有粗细，在某些场景下，我们更倾向于把几个操作组合在一起，看做一个更大粒度的算子。比如在入门时的手写数字识别 MNIST 中，`Linear` 和 `ReLU` 都是独立算子；而在 Transformer 架构中，我们把 `Linear + 激活函数 + Linear` 看成 MLP 整体，而 Attention 算子中，更是包含了一大堆复杂的计算。

3. **网络结构**：算子如何组织在一起，这就是网络结构。我们讨论网络结构也分为粒度，大的网络结构，或者说网络结构的风格，我一般更愿意称之为网络架构，典型的网络架构有 MLP（FNN），CNN，RNN及其变体（比如LSTM，GRU），GNN 以及 Transformer 结构。小的网络结构，是指模型子模块的连接方式，比如 Sequential、ModuleList。

### 实际的例子

下面是一个简单两层的 MLP 模型，`Linear + ReLU + Linear`，这里仅给了模型代码，没有展示训练的过程（数据加载，epoch迭代，mini-batch，正向传播 + 计算loss + 反向传播）。

```python
import torch
import torch.nn as nn

# Define a simple Feedforward Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create the model instance
input_size = 10    # example input feature size
hidden_size = 40   # hidden layer size
output_size = 2    # number of output classes
model = SimpleNN(input_size, hidden_size, output_size)
```

SimpleNN 是继承 nn.Module 的子类，在类初始化的第一件事情，是先调用 `nn.Module::__init__` 方法，它会初始化模块相关的数据结构。另外，这个模型的算子和网络结构都很简单，这里不在赘述。

那么，这个模型的参数在哪里呢？因为我们这里用的都是 torch.nn 中封装好的模块。我们打印出来看看：

```python
>>> print(model)
SimpleNN(
  (fc1): Linear(in_features=10, out_features=40, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=40, out_features=2, bias=True)
)
>>> model.state_dict().keys()
odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
>>> for name, param in model.named_parameters():
...     print(f"parameter {name}, param.shape={param.shape}, param.dtype={param.dtype}")
...
parameter fc1.weight, param.shape=torch.Size([40, 10]), param.dtype=torch.float32
parameter fc1.bias, param.shape=torch.Size([40]), param.dtype=torch.float32
parameter fc2.weight, param.shape=torch.Size([2, 40]), param.dtype=torch.float32
parameter fc2.bias, param.shape=torch.Size([2]), param.dtype=torch.float32
```

### state_dict

`model.state_dict()` 返回是模型持久化时存储到外部的张量的集合，它的命名规则是：
`算子名.子算子名......参数名`。比如 `fc1.weight`，fc1 是位于根模型的算子名，weight 是它在 Linear 这个算子中的参数名。如果某个算子是一个数组型的（Sequential 或者 ModuleList），则子算子名是数组索引。

通过 state_dict 的张量命名，可以看到模型的父子结构。

### torch.nn.Linear 的参数形状

让我们来分析 fc1 的 weight 和 bias 的参数形状。 Linear 算子对应的公式是 $y = X * A + b$, 让我们代入 input_features = 10, output_features = 40来看，$X$ 的形状是 $[S, 10]$ （其中 S 是样本个数）, $A$ 的形状是 $[10,40]$, $b$ 的形状为 $[40]$。

再看上面打印的参数， fc1.bias 完全可以对应 $b$。但是，fc1.weight 的形状是 $[40,10]$, 跟上面预期的 $[10,40]$ 是相反的，问题出现在哪里了？让我们再来看看 Linear 的文档：

> Applies a linear transformation to the incoming data: $y = xA^T + b$.

原因是因为 Linear 这个算子计算的是 $X * A^T$，而不是 $X * A$，所以需要将 $A$ 的形状做个转置，这下完美对应了。为什么需要这样子设计？原因也很简单，为了计算性能。

## 从参数文件出发，还原 GPT2 模型网络结构

让我们写一个简单的程序，把 GPT2 模型参数的结构打印出来：
```bash
h.0.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.0.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.0.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.0.attn.c_proj.bias , torch.float32 torch.Size([768])
h.0.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.0.ln_1.bias , torch.float32 torch.Size([768])
h.0.ln_1.weight , torch.float32 torch.Size([768])
h.0.ln_2.bias , torch.float32 torch.Size([768])
h.0.ln_2.weight , torch.float32 torch.Size([768])
h.0.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.0.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.0.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.0.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.1.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.1.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.1.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.1.attn.c_proj.bias , torch.float32 torch.Size([768])
h.1.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.1.ln_1.bias , torch.float32 torch.Size([768])
h.1.ln_1.weight , torch.float32 torch.Size([768])
h.1.ln_2.bias , torch.float32 torch.Size([768])
h.1.ln_2.weight , torch.float32 torch.Size([768])
h.1.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.1.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.1.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.1.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.10.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.10.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.10.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.10.attn.c_proj.bias , torch.float32 torch.Size([768])
h.10.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.10.ln_1.bias , torch.float32 torch.Size([768])
h.10.ln_1.weight , torch.float32 torch.Size([768])
h.10.ln_2.bias , torch.float32 torch.Size([768])
h.10.ln_2.weight , torch.float32 torch.Size([768])
h.10.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.10.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.10.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.10.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.11.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.11.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.11.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.11.attn.c_proj.bias , torch.float32 torch.Size([768])
h.11.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.11.ln_1.bias , torch.float32 torch.Size([768])
h.11.ln_1.weight , torch.float32 torch.Size([768])
h.11.ln_2.bias , torch.float32 torch.Size([768])
h.11.ln_2.weight , torch.float32 torch.Size([768])
h.11.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.11.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.11.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.11.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.2.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.2.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.2.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.2.attn.c_proj.bias , torch.float32 torch.Size([768])
h.2.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.2.ln_1.bias , torch.float32 torch.Size([768])
h.2.ln_1.weight , torch.float32 torch.Size([768])
h.2.ln_2.bias , torch.float32 torch.Size([768])
h.2.ln_2.weight , torch.float32 torch.Size([768])
h.2.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.2.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.2.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.2.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.3.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.3.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.3.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.3.attn.c_proj.bias , torch.float32 torch.Size([768])
h.3.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.3.ln_1.bias , torch.float32 torch.Size([768])
h.3.ln_1.weight , torch.float32 torch.Size([768])
h.3.ln_2.bias , torch.float32 torch.Size([768])
h.3.ln_2.weight , torch.float32 torch.Size([768])
h.3.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.3.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.3.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.3.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.4.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.4.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.4.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.4.attn.c_proj.bias , torch.float32 torch.Size([768])
h.4.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.4.ln_1.bias , torch.float32 torch.Size([768])
h.4.ln_1.weight , torch.float32 torch.Size([768])
h.4.ln_2.bias , torch.float32 torch.Size([768])
h.4.ln_2.weight , torch.float32 torch.Size([768])
h.4.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.4.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.4.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.4.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.5.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.5.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.5.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.5.attn.c_proj.bias , torch.float32 torch.Size([768])
h.5.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.5.ln_1.bias , torch.float32 torch.Size([768])
h.5.ln_1.weight , torch.float32 torch.Size([768])
h.5.ln_2.bias , torch.float32 torch.Size([768])
h.5.ln_2.weight , torch.float32 torch.Size([768])
h.5.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.5.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.5.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.5.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.6.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.6.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.6.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.6.attn.c_proj.bias , torch.float32 torch.Size([768])
h.6.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.6.ln_1.bias , torch.float32 torch.Size([768])
h.6.ln_1.weight , torch.float32 torch.Size([768])
h.6.ln_2.bias , torch.float32 torch.Size([768])
h.6.ln_2.weight , torch.float32 torch.Size([768])
h.6.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.6.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.6.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.6.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.7.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.7.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.7.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.7.attn.c_proj.bias , torch.float32 torch.Size([768])
h.7.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.7.ln_1.bias , torch.float32 torch.Size([768])
h.7.ln_1.weight , torch.float32 torch.Size([768])
h.7.ln_2.bias , torch.float32 torch.Size([768])
h.7.ln_2.weight , torch.float32 torch.Size([768])
h.7.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.7.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.7.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.7.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.8.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.8.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.8.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.8.attn.c_proj.bias , torch.float32 torch.Size([768])
h.8.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.8.ln_1.bias , torch.float32 torch.Size([768])
h.8.ln_1.weight , torch.float32 torch.Size([768])
h.8.ln_2.bias , torch.float32 torch.Size([768])
h.8.ln_2.weight , torch.float32 torch.Size([768])
h.8.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.8.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.8.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.8.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
h.9.attn.bias , torch.float32 torch.Size([1, 1, 1024, 1024])
h.9.attn.c_attn.bias , torch.float32 torch.Size([2304])
h.9.attn.c_attn.weight , torch.float32 torch.Size([768, 2304])
h.9.attn.c_proj.bias , torch.float32 torch.Size([768])
h.9.attn.c_proj.weight , torch.float32 torch.Size([768, 768])
h.9.ln_1.bias , torch.float32 torch.Size([768])
h.9.ln_1.weight , torch.float32 torch.Size([768])
h.9.ln_2.bias , torch.float32 torch.Size([768])
h.9.ln_2.weight , torch.float32 torch.Size([768])
h.9.mlp.c_fc.bias , torch.float32 torch.Size([3072])
h.9.mlp.c_fc.weight , torch.float32 torch.Size([768, 3072])
h.9.mlp.c_proj.bias , torch.float32 torch.Size([768])
h.9.mlp.c_proj.weight , torch.float32 torch.Size([3072, 768])
ln_f.bias , torch.float32 torch.Size([768])
ln_f.weight , torch.float32 torch.Size([768])
wpe.weight , torch.float32 torch.Size([1024, 768])
wte.weight , torch.float32 torch.Size([50257, 768])
```

根据上面对 state_dict() 的分析，不难还原出来：

![GPT2模型](/assets/2026-06-25/gpt2-model.png)

代码可见这里：

# Stay Curious

