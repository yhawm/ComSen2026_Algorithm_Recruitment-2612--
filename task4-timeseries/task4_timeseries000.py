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

# ================= 1. 基础配置 =================
os.makedirs('../output', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ================= 2. 数据加载与预处理 =================
# 自动下载经典的 Air Passengers 数据集（每月乘客数量）
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"网络下载失败，请手动下载 airline-passengers.csv 放到 task-4 目录下。错误: {e}")
    exit()

data = df['Passengers'].values.reshape(-1, 1).astype(np.float32)
print(f"数据形状: {data.shape}, 范围: [{data.min()}, {data.max()}]")

# 归一化 (将数据缩放到 0-1 之间，加速模型收敛)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 构建滑动窗口序列
# 用过去 12 个月的数据预测第 13 个月
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 12
X, y = create_sequences(data_scaled, SEQ_LEN)
print(f"序列数据: X shape={X.shape}, y shape={y.shape}")

# 划分训练集和测试集 (80% 训练，20% 测试)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ================= 3. 定义 LSTM 模型 =================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

model = LSTMPredictor().to(device)
print(model)

# ================= 4. 训练配置 =================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# ================= 5. 开始训练 =================
epochs = 100
train_losses, test_losses = [], []
best_test_loss = float('inf')

print("开始训练...")
for epoch in range(epochs):
    # 训练阶段
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

    # 验证阶段
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            test_loss += criterion(output, batch_y).item() * batch_x.size(0)
    test_loss /= len(test_loader.dataset)

    scheduler.step(test_loss)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), '../output/lstm_best.pth')

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# ================= 6. 测试与指标评估 =================
print("\n开始评估...")
model.load_state_dict(torch.load('../output/lstm_best.pth'))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        all_preds.extend(output.cpu().numpy().flatten())
        all_true.extend(batch_y.numpy().flatten())

# 反归一化，还原成真实乘客数量
all_preds = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
all_true = scaler.inverse_transform(np.array(all_true).reshape(-1, 1)).flatten()

# 计算指标
mae = mean_absolute_error(all_true, all_preds)
mse = mean_squared_error(all_true, all_preds)
rmse = np.sqrt(mse)

print(f"\n===== 评估指标 =====")
print(f"MAE (平均绝对误差): {mae:.2f}")
print(f"MSE (均方误差): {mse:.2f}")
print(f"RMSE (均方根误差): {rmse:.2f}")

# ================= 7. 可视化结果 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：Loss 曲线
axes[0].plot(train_losses, 'b-', label='Train Loss')
axes[0].plot(test_losses, 'r-', label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training & Test Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：预测对比图
axes[1].plot(range(len(all_true)), all_true, 'b-', label='True Values', alpha=0.7)
axes[1].plot(range(len(all_preds)), all_preds, 'r--', label='Predicted Values', alpha=0.7)
axes[1].set_xlabel('Time Step (Months)')
axes[1].set_ylabel('Passengers')
axes[1].set_title(f'Prediction vs Truth (RMSE={rmse:.2f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/timeseries_result.png', dpi=150)
plt.show()
print("\n图像已保存到 ../output/timeseries_result.png")