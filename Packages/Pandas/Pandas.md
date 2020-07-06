# Pandas

### 数据结构

* DataFrame
* Series

#### DataFrame

|      | 备注   | 工资 | 绩效分 |
| ---- | ------ | ---- | ------ |
| 小刘 | 不及格 | 5000 | 60     |
| 小张 | 良好   | 7000 | 80     |
| 小赵 | 优秀   | 9000 | 95     |

```python
# dataFrame为pandas中的表结构, 可理解为excel中的一张表

# 0.导入pandas库
import pandas as pd

# 1.创建: 字典+列表
df = pd.DataFrame({'工资':[5000,7000,9000],'绩效分':[60,80,95],'备注':['不及格','良好','优秀']},
                 index = ['小刘','小张','小赵'])
# 字典: 每一对 key-values 对应表中一列, key对应列标题, values对应列值
# 列表: 最左侧的index, 若创建时不指定, 则自动生成从0开始的索引

# 2.读取: 读取 CSV 格式的文件, 或者读取 EXCEL 格式(.xlsx和xls后缀)的文件
df = pd.read_csv("XXX.csv", engine='python')		# 读取CSV文件
df = pd.read_excel('XXX.xls')		# 读取Excel文件

# 3.存储: 存储格式为CSV/Excel
df.to_csv('XXX.csv')		# 存储为csv格式
df.to_excel('XXX.xlsx')	# 存储为excel格式

# 4.查看头部/尾部数据
df.head()		# 可查看默认的前5行, 指定参数x可查看前x行: df.head(x)
df.tail()		# 可查看默认的后5行, 指定参数x可查看后x行: df.tail(x)

# 5.格式查看
df.info()		# 查看数据集的行列数,列的数据类型以及缺失情况

# 6.统计信息概览
df.describe()		# 计算数值型数据的关键统计指标: 平均数, 中位数, 标准差等

# 7.列的基本处理方式 -- 增, 删, 改, 选
df['新增的列'] = [1, 2, 3, 4, 5]		# 增加一列, df['新列'] = 新列值
df.drop('新增的列', axis = 1, inplace = True)	# axis=1对列操作, inplace为True则在源数据上修改
df['第一列']		# 选取指定的一列
df[['第一列','第二列']]		# 选取指定的多列



# 8.常用数据类型及操作
# 字符串
# 数值型
# 时间类型
```





