# 实验环境

## 系统环境

+ Ubuntu 20.04
+ RTX 4090

## 所需依赖

+ PyTorch==2.0.0 + CUDA\==11.8
+ torchvision
+ logging
+ matplotlib
+ random

# 数据集

CIFAR_100 [官方地址](https://www.cs.toronto.edu/~kriz/cifar.html)

不需要手动下载，运行代码会自动在 `./data` 中自动下载

# 运行方式

文件名即为所用优化器，在终端输入 `python [filename]` 即可运行

# 实验结果

| Optimizer    | Accuracy | Time    |
| ------------ | -------- | ------- |
| SGD          | 44.62%   | 53’18’’ |
| Adam         | 54.45%   | 53’28’’ |
| SGD+SAM      | 53.44%   | 65’46’’ |
| Adam+SAM     | 55.60%   | 69’01’’ |
| SGD+ESAM     | 52.80%   | 54’17’’ |
| Adam+ESAM    | 54.88%   | 58’34’’ |
| SGD+LookSAM  | 56.45%   | 54’33’’ |
| Adam+LookSAM | 53.99%   | 55’34’’ |