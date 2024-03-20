import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import chardet

# Path to the test data and the model file
test_data_path = 'gan_str_te.csv'  # 测试文件路径
model_file_path = 'FTTransformer.pth'  # 模型路径

# Explicitly setting the device to CPU
device = torch.device("cpu")

# Load the model ensuring map_location is explicitly targeting CPU
model = torch.load(model_file_path, map_location=device)
model.to(device)
model.eval()

# Load and preprocess the test data
with open(test_data_path, 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
print(encoding)
test_data = pd.read_csv(test_data_path, encoding=encoding)
X_test = test_data.iloc[:, :-1].values

# Preprocess the data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # Ideally, use the scaler from the training phase

# Convert to PyTorch tensors and move to the specified device (CPU)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# Create a DataLoader for the test set
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=8000)  # Adjust the batch size as needed

# Predict using the model
predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        outputs = model(X_batch[0], None).squeeze()  # Ensure your model forward method is compatible
        predicted = torch.sigmoid(outputs).round()
        predictions.extend(predicted.cpu().numpy())

# Convert X_test_scaled back to a DataFrame for easy concatenation
X_test_df = pd.DataFrame(X_test_scaled, columns=test_data.columns[:-1])

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Predicted'])

# Concatenate X_test_df and predictions_df
final_df = pd.concat([X_test_df, predictions_df], axis=1)

# Save the concatenated DataFrame to CSV
final_csv_path = 'model_predictions_with_X_test.csv'#生成的预测文件路径
#final_df.to_csv(final_csv_path, index=False)
#print(f'Predictions with X_test variables saved to {final_csv_path}')
final_df.to_csv(final_csv_path, index=False, encoding='utf-8')  # Specify encoding as 'utf-8'
#final_df.to_csv('model_predictions_with_X_test_gbk.csv', index=False, encoding='gbk')
print(f'Predictions with X_test variables saved to {final_csv_path} in UTF-8 encoding.')