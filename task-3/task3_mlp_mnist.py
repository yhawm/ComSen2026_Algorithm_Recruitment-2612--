import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import io
from PIL import Image

# ========== 设备配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 【终极修复版】Dataset ==========
class ParquetMNIST(Dataset):
    def __init__(self, parquet_path):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"找不到文件: {parquet_path}")
        
        self.df = pd.read_parquet(parquet_path)
        print(f"成功加载 {parquet_path}，数据集大小: {len(self.df)}")
        print(f"数据集列名: {self.df.columns.tolist()}")
        
        self.label_col = 'label'
        self.pixel_col = 'image'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 提取标签
        label = self.df.iloc[idx][self.label_col]
        
        # 2. 提取图像数据
        img_data = self.df.iloc[idx][self.pixel_col]
        
        # 3. 核心解码逻辑：根据数据类型不同采取不同操作
        if isinstance(img_data, dict):
            # 情况A：数据是字典，包含 'bytes' 字段（如 HuggingFace 数据集）
            if 'bytes' in img_data:
                # 用 PIL 从内存字节流中解码图片
                image = Image.open(io.BytesIO(img_data['bytes'])).convert('L')
                pixels = np.array(image, dtype=np.float32).flatten()
            # 情况B：数据是字典，包含 'array' 字段
            elif 'array' in img_data:
                pixels = np.array(img_data['array'], dtype=np.float32).flatten()
            else:
                raise ValueError(f"未知的字典格式，键为: {img_data.keys()}")
        elif isinstance(img_data, (np.ndarray, list)):
            # 情况C：数据已经是数组或列表
            pixels = np.array(img_data, dtype=np.float32).flatten()
        else:
            raise TypeError(f"不支持的图像数据类型: {type(img_data)}")
        
        # 4. 尺寸检查
        if pixels.size != 784:
            raise ValueError(f"图像数据尺寸异常，期望784，实际{pixels.size}。")
        
        # 5. 归一化
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        
        # 6. 转为 Tensor 并 reshape
        img_tensor = torch.tensor(pixels).reshape(1, 28, 28)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

# ========== 数据加载与预处理 ==========
parquet_path = 'train-00000-of-00001.parquet' 
full_dataset = ParquetMNIST(parquet_path=parquet_path)

# 手动切分 70% / 15% / 15%
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

# ========== 定义 MLP 模型 ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = MLP().to(device)
print(model)

# ========== 训练配置 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

os.makedirs('../output/task3', exist_ok=True)

# ========== 训练循环 ==========
epochs = 15
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

for epoch in range(epochs):
    # 训练
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss, train_acc = running_loss / total, correct / total

    # 验证
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_loss, val_acc = val_running_loss / val_total, val_correct / val_total
    scheduler.step()

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc); val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../output/task3/best_mlp_model.pth')

# ========== 测试 ==========
model.load_state_dict(torch.load('../output/task3/best_mlp_model.pth'))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"\n===== 测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%) =====")

# ========== 绘图与可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, epochs+1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, epochs+1), val_losses, 'r-o', label='Val Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss Curve'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, epochs+1), train_accs, 'b-o', label='Train Acc')
axes[1].plot(range(1, epochs+1), val_accs, 'r-o', label='Val Acc')
axes[1].axhline(y=test_acc, color='g', linestyle='--', label=f'Test Acc: {test_acc:.4f}')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/task3/mnist_loss_acc.png', dpi=150)
plt.show()

model.eval()
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)
with torch.no_grad():
    outputs = model(test_images[:6])
    _, preds = torch.max(outputs, 1)

fig2, axes2 = plt.subplots(1, 3, figsize=(10, 4))
for i in range(3):
    img = test_images[i].cpu().squeeze().numpy()
    pred = preds[i].item()
    true = test_labels[i].item()
    axes2[i].imshow(img, cmap='gray')
    axes2[i].set_title(f'Pred: {pred} | True: {true}', color='green' if pred == true else 'red')
    axes2[i].axis('off')
plt.tight_layout()
plt.savefig('../output/task3/mnist_visualization.png', dpi=150)
plt.show()

print("所有图像已保存到 ../output/task3/ 目录")
