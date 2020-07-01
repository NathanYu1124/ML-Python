# NumPy

### NumPy介绍

NumPy是Python中科学计算的基础包。它是一个Python库，提供多维数组对象，各种派生对象（如掩码数组和矩阵），以及用于数组快速操作的各种API。NumPy包的核心是 ***ndarray*** 对象。它封装了python原生的同数据类型的 *n* 维数组，为了保证其性能优良，其中有许多操作都是代码在本地进行编译后执行的。



#### NumPy数组和原生Python Array之间的重要的区别：

* NumPy 数组在创建时具有固定的大小，与Python的原生数组对象（可以动态增长）不同。更改ndarray的大小将创建一个新数组并删除原来的数组。
* NumPy 数组中的元素都需要具有相同的数据类型，因此在内存中的大小相同。 例外情况：Python的原生数组里包含了NumPy的对象的时候，这种情况下就允许不同大小元素的数组。
* NumPy 数组有助于对大量数据进行高级数学和其他类型的操作。通常，这些操作的执行效率更高，比使用Python原生数组的代码更少。



#### NumPy的特征: 矢量化

```python
# 矢量化描述了代码中没有任何显式的循环，索引等 - 这些当然是预编译的C代码中“幕后”优化的结果。
# 矢量化代码有许多优点，其中包括：
# 1.矢量化代码更简洁，更易于阅读
# 2.更少的代码行通常意味着更少的错误
# 3.代码更接近于标准的数学符号（通常，更容易正确编码数学结构）
# 4.矢量化导致产生更多 “Pythonic” 代码。如果没有矢量化，代码就会被低效且难以阅读的for循环所困扰。


# 例: 两个一维/二维数组a, b, 将其对应元素相乘保存在c中, 使用NumPy时a,b,c均为ndarray

# Python代码: python中循环效率低下, 数据过多时代码执行效率低
c = []
for i in range(len(a)):
  c.append(a[i] * b[i])
 
# C代码: c的高效性解决了python的循环效率低下的问题
for(int i = 0; i < rows; i++): {
  c[i] = a[i] * b[i];
}
  
# NumPy代码: 兼顾python的简洁与c的高效, ndarray的逐个元素的操作由预编译的c代码执行
c = a * b

```



#### NumPy的特征: 广播

广播是用于描述操作的隐式逐元素行为的术语。一般来说，在NumPy中，所有操作，不仅仅是算术运算，而是逻辑，位，功能等，都以这种隐式的逐元素方式表现，即它们进行广播。此外，在上面的例子中，`a`并且`b`可以是相同形状的多维数组，或者标量和数组，或者甚至是具有不同形状的两个数组，条件是较小的数组可以“扩展”到更大的形状。



### NumPy基础

#### 数据类型

| NumPy类型(dtype) | C类型        |
| ---------------- | ------------ |
| np.bool          | bool         |
| np.byte          | char         |
| np.short         | short        |
| np.int           | int          |
| np.uint          | unsigned int |
| np.longlong      | long long    |
| np.float         | float        |
| np.double        | double       |



#### 创建数组

```python
# 创建NumPy数组的最重要方法:
# 1.从Python结构（例如，列表，元组）转换
# 2.numpy原生数组的创建（例如，arange、ones、zeros等）

# 0.numpy数组的常用属性:
# -> ndim(维度)
# -> shape(行数和列数)
# -> size(元素个数)

# 1.将Python array_like对象转换为Numpy数组, 通常使用array()函数将其转换为Numpy数组
x = np.array([2,3,1,0])		# 用list创建


# 2.Numpy原生数组的创建 
np.zeros((2, 3))			#	创建数据全为0的数组, 参数:(shape, dtype, order)
np.ones((2, 3))				#	创建数据全为1的数组, 参数:(shape, dtype, order)
np.empty((3, 5))			# 创建数据接近0的数组, 参数:(shape, dtype, order)
np.arange(10,20,2)		#	创建数据为指定范围按步长增长的数组, 参数:(start, stop, step, dtype) 
np.linspace(1,4,6)		# 创建指定数量并在范围内平均间隔的数组，
											# 参数:(start,stop,num,endpoint,retstep,dtype,axis))
np.indices((3,3))			# 参数:(dimensions, dtype=int, sparse=False)

```



#### NumPy基本运算

```python
a = np.array([10,20,30,40])		# array([10, 20, 30, 40])
b = np.arange(4)							# array([0, 1, 2, 3])
c = np.array([[1,2], [4,5]])	# array([[1,2], [4,5]])

# 1.加法运算
c = a + b			# array([10, 21, 32, 43])

# 2.减法运算
c = a - b		  # array([10, 19, 28, 37])
	
# 3.乘法运算: 对应元素相乘
c = a * b 		# array([0, 20, 60, 120])

# 4.乘法运算: 矩阵乘法运算
c = np.dot(a, b)		# 行向量*列向量, 200

# 5.乘方运算	
c = b ** 2		# array([0, 1, 4, 9])

# 矩阵所有元素求和
np.sum(a)			# 100

# 矩阵所有元素寻找最小值
np.min(a)			# 10

# 矩阵所有元素寻找最大值
np.max(a)			# 40

# 矩阵的均值
np.mean(a)			# 25.0
np.average(a)		# 25.0

# 矩阵的累加之和: 每一项矩阵元素都是从原矩阵首项累加到对应项的元素之和
np.cumsum(a)		# [10, 30, 60, 100]

# 矩阵转置
np.transpose(c)
c.T
```



#### NumPy索引

```python
a = np.array([3, 4, 5, 6, 7, 8, 9, 10])
b = np.array([[3, 4, 5, 6], [7, 8, 9, 10]])

# 一维索引: [0, len(a) - 1]
a[3]			# 6
b[1]			# [7, 8, 9, 10]

# 二维索引
b[0][1]		# 4
b[0, 1]		# 4
b[0, 1:3]	# [4, 5], : 切片操作

```



#### NumPy数组合并

```python
a = np.array([1, 1, 1])
b = np.array([2, 2, 2])

# 上下合并
np.vstack(a, b)		# [[1,1,1], [2,2,2]]

# 左右合并
np.hstack(a, b)		# [1,1,1,2,2,2]

# 行/列的转置 -> 矩阵: np.newaxis()
c = a[np.newaxis, :]	# [[1,1,1]], 				1X3的矩阵
c = a[:, np.newaxis]	# [[1], [1], [1]],	3X1的矩阵

# 合并多个矩阵或序列: np.concatenate()
A = np.array([1,1,1])[:,np.newaxis]		# 3X1的矩阵
B = np.array([2,2,2])[:,np.newaxis]		# 3X1的矩阵

c = np.concatenate((A,B,A), axis=0)		# axis = 0,纵向合并, axis = 1, 横向合并

```



#### NumPy数组分割

```python



```



























































