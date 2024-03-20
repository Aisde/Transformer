import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import chardet

# Path to the test data and the model file
test_data_path = 'gan_str_te.csv'  # 这里选择测试集文件的路径
model_file_path = 'FTTransformer.pth'  # 模型路径

# 显式设置设备为CPU
device = torch.device("cpu")

# 加载模型，确保显式使用map_location指向CPU
model = torch.load(model_file_path, map_location=device)
model.to(device)
model.eval()

# Load and preprocess the test data
with open(test_data_path, 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
test_data = pd.read_csv(test_data_path, encoding=encoding)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Preprocess the data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # Ideally, use the scaler from the training phase

# Convert to PyTorch tensors and move to the specified device (CPU)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create a DataLoader for the test set
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=8000)  # Adjust the batch size as needed

# Test the model
correct = 0
total = 0
predictions = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch, None).squeeze()  # Ensure your model forward method is compatible

        predicted = torch.sigmoid(outputs).round()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        # Store predictions
        predictions.extend(predicted.cpu().numpy())
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
# Convert predictions to the same shape as y_test for classification report
predictions = np.array(predictions).flatten()

# Print classification report
print(classification_report(y_test, predictions))