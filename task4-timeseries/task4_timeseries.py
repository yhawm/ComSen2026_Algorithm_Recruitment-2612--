import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

os.makedirs('../output/task4-timeseries', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 1. 数据加载与预处理 ==========
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"在线下载失败，尝试本地读取: {e}")
    df = pd.read_csv('passengers.csv')

data = df['Passengers'].values.reshape(-1, 1).astype(np.float32)

# Min-Max 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 【回退】窗口大小改回 12
SEQ_LEN = 12  
X, y = create_sequences(data_scaled, SEQ_LEN)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 【回退】Batch size 调小
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=8, shuffle=False)

# ========== 2. 定义更简单的 LSTM 模型 ==========
class SimpleLSTMPredictor(nn.Module):
    # 【回退】隐藏层降回 64，单层 LSTM
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 简化全连接层，直接输出，避免过度非线性拟合
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = SimpleLSTMPredictor().to(device)

# ========== 3. 训练配置（去掉强正则化） ==========
criterion = nn.MSELoss()
# 【回退】取消 weight_decay，适当提高学习率，让模型能跳动起来
optimizer = optim.Adam(model.parameters(), lr=0.005) 

epochs = 100
train_losses, test_losses = [], []
best_test_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            test_loss += criterion(output, batch_y).item() * batch_x.size(0)
    test_loss /= len(test_loader.dataset)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), '../output/task4-timeseries/lstm_best.pth')

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# ========== 4. 模型评估与反归一化 ==========
model.load_state_dict(torch.load('../output/task4-timeseries/lstm_best.pth'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        all_preds.extend(output.cpu().numpy().flatten())
        all_true.extend(batch_y.numpy().flatten())

all_preds = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
all_true = scaler.inverse_transform(np.array(all_true).reshape(-1, 1)).flatten()

mae = mean_absolute_error(all_true, all_preds)
mse = mean_squared_error(all_true, all_preds)
rmse = np.sqrt(mse)

print(f"\n===== 调整后的评估指标 =====")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# ========== 5. 绘图 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, 'b-', label='Train Loss')
axes[0].plot(test_losses, 'r-', label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training & Test Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(len(all_true)), all_true, 'b-', label='True Values', alpha=0.8)
axes[1].plot(range(len(all_preds)), all_preds, 'r--', label='Predicted Values', alpha=0.8)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Passengers')
axes[1].set_title(f'Prediction vs Truth (RMSE={rmse:.2f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/task4-timeseries/timeseries_result.png', dpi=150)
plt.show()

print("结果图已保存到 ../output/task4-timeseries/timeseries_result.png")