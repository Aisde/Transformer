import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
# 1. 准备数据
#iris = load_iris()
#X = iris.data  # 特征数据
#y = iris.target  # 目标数据
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
# 由于鸢尾花数据集的目标标签已经是数值型，无需LabelEncoder转换

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 特征数据标准化

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# 4. 定义模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 5. 初始化模型和超参数
input_size = X.shape[1]  # 特征数
hidden_size = 64  # 隐藏层大小
num_classes = len(set(y))  # 类别数
learning_rate = 0.0001
num_epochs = 10000

model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 训练模型
for epoch in range(num_epochs):
    # 将训练数据转换为PyTorch张量
    inputs = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(y_train).long()

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每隔一段时间打印损失值
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 7. 在测试集上评估模型
model.eval()
with torch.no_grad():
    inputs = torch.from_numpy(X_test).float()
    labels = torch.from_numpy(y_test).long()

    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == labels).sum().item() / len(labels)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(classification_report(y_test,  predicted.cpu().numpy()))
# 8. 保存模型
#torch.save(model.state_dict(), 'mlp_model_iris.pth')
#print("Model saved")
