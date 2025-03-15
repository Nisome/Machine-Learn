from numpy import *  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import pandas as pd  # 导入pandas库，用于数据处理
from numpy.linalg import *  # 导入numpy的线性代数模块

# 读取iris数据集，文件名为'iris.data.csv'，没有表头
df = pd.read_csv('iris.data.csv', header=None)

# 将最后一列的类别名称映射为数字
df[4] = df[4].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# 打印数据集的前几行，查看数据是否正确加载
print(df.head())

# 提取特征数据（前四列），并将其转换为numpy数组
data = df.iloc[:, :-1].values

# 获取数据的样本数和特征数
samples, features = df.shape

# 设置降维后的维度为2
S = 2

# 对数据进行奇异值分解（SVD），得到U, s, V三个矩阵
U, s, V = linalg.svd(data)

# 构建对角矩阵Sig，只保留前S个奇异值
Sig = mat(eye(S) * s[:S])

# 打印降维后的奇异值矩阵
print(Sig)

# 使用U矩阵的前S列作为降维后的数据
newdata = U[:, :S]

# 创建一个图形对象
fig = plt.figure()

# 在图形中添加一个子图
ax = fig.add_subplot(1, 1, 1)

# 定义不同类别的标记样式
marks = ['o', '^', '+']

# 遍历每个样本，绘制散点图，并设置x轴和y轴的标签
for i in range(samples):
    ax.scatter(newdata[i, 0], newdata[i, 1], c='black', marker=marks[int(data[i, -1])])

plt.xlabel('SVD1')
plt.ylabel('SVD2')

# 显示图形
plt.show()