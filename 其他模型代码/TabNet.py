from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
# 加载鸢尾花数据集
#data = load_iris()
#X = data.data
#y = data.target
import chardet
#data = pd.read_csv(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv')
with open(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv', 'rb') as f:
    result = chardet.detect(f.read())  # 读取文件的一部分用于检测

print(result['encoding'])
# 然后使用检测到的编码打开文件
data = pd.read_csv(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv', encoding=result['encoding'])
#save_dir=r'C:\Users\你的时间\Desktop\深度学习论文\model\model3.pth'#保存路径
X=data.iloc[:,0:-1]
y=data.iloc[:,-1]
X = np.array(X)
y = np.array(y)
# 将数据集分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 标准化特征数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")
# 实例化TabNet分类器
clf = TabNetClassifier(verbose=0, seed=42)

# 训练模型
clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    max_epochs=100,  # 可以调整为更合适的epoch数量
    patience=10,  # 用于early stopping
    batch_size=8000,  # 可以根据你的数据和显存大小调整
    virtual_batch_size=8  # 可以根据你的数据和显存大小调整
)

# 使用训练好的模型进行预测
preds = clf.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, preds))