# 封装2.0  OK
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from rtdl_revisiting_models import FTTransformer
import argparse
import chardet


def train_and_evaluate(train_dir, test_dir, lr, n_epochs, batch_size):
    # Load and preprocess the training data
    with open(train_dir, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    data = pd.read_csv(train_dir, encoding=encoding)
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Load and preprocess the test data
    with open(test_dir, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    test_data = pd.read_csv(test_dir, encoding=encoding)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    X_test_scaled = scaler.transform(X_test)

    # Move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
###y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
#y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Define the model, loss, and optimizer
    model = FTTransformer(
        n_cont_features=X_train_tensor.shape[1],
        cat_cardinalities=[],
        d_out=1,
        n_blocks=3,
        d_block=192,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden_multiplier=4,
        ffn_dropout=0.1,
        residual_dropout=0.2,
        linformer_kv_compression_ratio=0.2,
        linformer_kv_compression_sharing='headwise'
    ).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Train the model
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch, None).squeeze()
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor, None).squeeze().sigmoid().round()
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test Accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate FTTransformer')
    parser.add_argument('--train_dir', type=str, help='训练集路径Path to the training data CSV file', required=False,
                        default=r'gan_str_tr.csv')
    parser.add_argument('--test_dir', type=str, help='测试集路径Path to the testing data CSV file', required=False,
                        default=r'gan_str_te.csv')
    parser.add_argument('--lr', type=float, help='学习率Learning rate for the optimizer', required=False, default=0.001)
    parser.add_argument('--n_epochs', type=int, help='迭代数Number of epochs to train', required=False, default=100)
    parser.add_argument('--batch_size', type=int,  help='批量Batch size for training',default=8000)
    args = parser.parse_args()

    train_and_evaluate(args.train_dir, args.test_dir, args.lr, args.n_epochs,args.batch_size)