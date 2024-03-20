import pandas as pd
from sklearn.linear_model import SGDClassifier
import numpy as np

# 再次导入数据以便清晰展示
data = pd.read_excel('./data/data.csv')  # 确保此路径正确
X = data.iloc[:, :-1].values  # 特征变量
y = data.iloc[:, -1].values   # 目标变量

# 使用弹性网正则化初始化SGD分类器
clf = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.1, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X, y)

# 获取模型系数
coefficients = clf.coef_.flatten()

# 计算系数的绝对值以确定重要性
importance = np.abs(coefficients)

# 累积重要性以选出贡献度达90%的特征
cumulative_importance = np.cumsum(importance[np.argsort(importance)[::-1]]) / np.sum(importance)
top_90_idx = np.where(cumulative_importance <= 0.9)[0]
selected_features_idx = np.argsort(importance)[::-1][top_90_idx]

# 获取这些特征的名称
feature_names = np.array(data.columns[:-1])  # 假设最后一列是标签
selected_features = feature_names[selected_features_idx]

# 获取对应的系数
selected_coefficients = coefficients[selected_features_idx]

# 打印被选择的特征和对应的系数
for feature, coeff in zip(selected_features, selected_coefficients):
    print(f"{feature}: {coeff}")

# 检查标签列的实际名称
label_column_name = data.columns[-1]  # 这应该是标签列的名称

# 如果标签列名称不是'label',请确保使用正确的标签列名称
data_new = data[np.append(selected_features, label_column_name)]

# 将新数据集保存到CSV文件中,确保文件路径是您希望保存的位置
data_new.to_csv('./data/data_feather.csv', index=False)
