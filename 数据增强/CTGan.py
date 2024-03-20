#能运行的GAN
from sklearn.datasets import load_iris
import pandas as pd
from ctgan import CTGAN
import numpy as np
# 加载数据集

data=pd.read_csv(r'data.csv')
X=data.iloc[:,0:-1]
y=data.iloc[:,-1]
X = np.array(X)
y = np.array(y)
# 将特征和目标变量合并到一个DataFrame中
# 注意：CTGAN主要设计用于处理DataFrame类型的数据
columns = list(data.columns)
data = pd.DataFrame(data=np.column_stack((X,y)), columns=columns)

# 指定目标列作为离散列，以符合CTGAN的使用方式
#discrete_columns = ['target']
discrete_columns = ['target'] if 'target' in columns else []
# 实例化CTGAN模型
ctgan = CTGAN(epochs=10)  # 可以根据需要调整epochs

# 训练模型
ctgan.fit(data, discrete_columns)

# 生成合成数据,按照需求修改生成的数据量
synthetic_data = ctgan.sample(16000)

# 显示生成的合成数据
print(synthetic_data.head())
# 保存生成的合成数据到 CSV 文件
synthetic_data.to_csv(r'synthetic_data.csv', index=False)
