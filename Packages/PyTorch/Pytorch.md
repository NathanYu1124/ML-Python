## Pytorch Basics

#### 1.Tensor - 张量

```python
# PyTorch是一个优化的张量操作库，这个库的核心就是张量(tensor), 它是一个包含一些多维数据的数学对象。
# 0阶张量就是一个数字, 或者称为标量
# 1阶张量是一个数字数组, 或者称为向量
# 2阶张量是一个向量数组, 或者称为矩阵
# 3阶张量是一个矩阵数组, 或者称为立方体
# n阶张量是一个n-1阶张量的数组 ？？？
```

| Data type                | dtype                           | CPU tensor         | GPU tensor              |
| ------------------------ | ------------------------------- | ------------------ | ----------------------- |
| 32-bit floating point    | torch.float32` or `torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.float64` or `torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point    | torch.float16` or `torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned) | torch.uint8                     | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8-bit integer (signed)   | torch.int8                      | torch.CharTensor   | torch.cuda.CharTensor   |
| 16-bit integer (signed)  | torch.int16` or `torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32-bit integer (signed)  | torch.int32` or `torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64-bit integer (signed)  | torch.int64` or `torch.long     | torch.LongTensor   | torch.cuda.LongTensor   |
| Boolean                  | torch.bool                      | torch.BoolTensor   | torch.cuda.BoolTensor   |



#### 2.创建张量

```python
# torch.Tensor 是tensor默认类型torch.FloatTensor的别名
# 任何带有下划线的PyTorch方法都是指就地(in place)操作, 在原tensor上修改内容

# 1.用存在的数据创建tensor
torch.tensor()

# 2.创建具体维度的tensor
torch.* 

# 3.创建与另一个tensor类型和维度都相同的tensor
torch.*_like

# 4.创建与另一个tensor类型相同但维度不同的tensor
tensor.new_*

torch.zeros([3, 2])		# 创建全0tensor
torch.ones([3, 2])		# 创建全1tensor
.fill_(5)							# 将tensor变量中的值全填充为指定值, 例如5
torch.rand(2, 3)			# tensor值为[0,1)的均匀分布中的一组随机数
torch.randn(2, 3)			# tensor值为标准正态分布(均值为0, 方差为1)中抽取的一组随机数

```













































































