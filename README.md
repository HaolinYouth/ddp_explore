# 多GPU训练
- 使用MNIST数据作为尝试 进行分布式训练
  
## 单卡训练
python DDP02.py -n 1 -g 1 -nr 0

## 多卡训练
python DDP03.py -n 1 -g 4 -nr 0

## 参考
部分代码来自于: [pytorch-分布式训练极简体验](https://zhuanlan.zhihu.com/p/477073906)和 [PyTorch分布式训练简明教程](https://blog.csdn.net/xiaohu2022/article/details/105325610)