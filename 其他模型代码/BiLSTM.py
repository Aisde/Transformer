import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')

##  读取数据

import chardet
#data = pd.read_csv(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv')
with open(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv', 'rb') as f:
    result = chardet.detect(f.read())  # 读取文件的一部分用于检测

print(result['encoding'])
# 然后使用检测到的编码打开文件
data = pd.read_csv(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv', encoding=result['encoding'])
feature_data = data.iloc[:, 0:-1]
label_data = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(feature_data,label_data, test_size=0.2,random_state=42)



def get_train_data():

    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)

    import pandas as pd
    from sklearn import preprocessing
    # 生成训练数据x并做标准化后，构造成dataframe格式，再转换为tensor格式

    DF = pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(X_train))
    y = pd.Series(Y_train)
    return get_tensor_from_pd(DF).float(), get_tensor_from_pd(y).float()

def get_test_data(X_test_in,Y_test_in):
    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.tensor(data=dataframe_series.values)

    import pandas as pd
    from sklearn import preprocessing
    # 生成训练数据x并做标准化后，构造成dataframe格式，再转换为tensor格式

    Df = pd.DataFrame(data=preprocessing.StandardScaler().fit_transform(X_test_in))
    y0 = pd.Series(Y_test_in)
    return get_tensor_from_pd(Df).float(), get_tensor_from_pd(y0).float()


class LSTM(nn.Module):
    def __init__(self, input_size=350, hidden_layer_size=200, output_size=1, num_layers=1):
        """
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size:隐层的数目
        :param output_size: 输出的个数
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_layer_size, num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size*2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        lstm_out, _ = self.bilstm(input_x)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # =self.linear(lstm_out[:, -1, :])
        predictions = self.sigmoid(linear_out)
        return predictions


if __name__ == '__main__':
    # 得到数据
    x, y = get_train_data()
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(x, y),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=70,  # 每块的大小
        shuffle=True,  # 要不要打乱数据
        num_workers=2,  # 多进程（multiprocess）来读数据
    )
    # 建模三件套：loss，优化，epochs
    model = LSTM()  # 模型
    loss_function = nn.BCELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 100
    # 开始训练
    model.train()
    list_ = []
    for i in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            print("Train Step:", i, " loss: ", single_loss)
            running_loss = single_loss.item()
            list_.append(running_loss)

    s=list(range(1,len(list_)+1))
    #plt.plot(s,list_)
    #plt.show()

    # 开始验证
    from sklearn.metrics import accuracy_score, classification_report

    # 开始验证
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        test_set_x, test_set_y = get_test_data(X_test, Y_test)
        y_set_pred = model(test_set_x).squeeze()
        # 将概率转换为类别标签
        y_set_pred_labels = (y_set_pred > 0.5).int()

    # 计算准确率
    accuracy = accuracy_score(test_set_y.numpy(), y_set_pred_labels.numpy())
    print(f'Accuracy: {accuracy:.4f}')

    # 输出分类报告
    print(classification_report(test_set_y.numpy(), y_set_pred_labels.numpy()))
