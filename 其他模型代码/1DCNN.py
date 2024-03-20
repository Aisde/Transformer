import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
# 导入数据
import chardet
#data = pd.read_csv(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv')
with open(r':\Users\你的时间\Desktop\深度学习论文\v2_tr.csv', 'rb') as f:
    result = chardet.detect(f.read())  # 读取文件的一部分用于检测

print(result['encoding'])
# 然后使用检测到的编码打开文件
data = pd.read_csv(r'C:\Users\你的时间\Desktop\深度学习论文\v2_tr.csv', encoding=result['encoding'])
feature_data = data.iloc[:, 0:-1]
label_data = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.25, random_state=42)

X = np.array(x_train)
y = np.array(y_train)

# 检查特征数量
feature_count = feature_data.shape[1]
print("特征数量:", feature_count)

# 数据预处理
X_1 = X.reshape(-1, feature_count, 1)  # 使用实际的特征数量重塑数据
y_1 = tf.keras.utils.to_categorical(y, num_classes=2)  # 将标签进行独热编码

# 创建1DCNN模型
model = models.Sequential()
model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(feature_count, 1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))  # 输出层

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_1, y_1, epochs=30, batch_size=32, validation_split=0.2)

# 评估模型
X_test = np.array(x_test)
y_test = np.array(y_test)
X_test_1 = X_test.reshape(-1, feature_count, 1)  # 使用实际的特征数量重塑数据
y_test_1 = tf.keras.utils.to_categorical(y_test, num_classes=2)
loss, accuracy = model.evaluate(X_test_1, y_test_1)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
# 注意这里不需要将y_test转换为独热编码
#model.eval()  # 如果使用的是tf.keras模型，这一行不是必须的
y_pred_probs = model.predict(X_test_1)
y_pred = np.argmax(y_pred_probs, axis=1)  # 从概率到类别标签的转换

# 现在y_test和y_pred都是类别标签的形式，我们可以生成classification_report
print(classification_report(y_test, y_pred))