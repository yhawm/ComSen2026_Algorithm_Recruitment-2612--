import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 1. 基础配置 =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 确保 output 目录存在
os.makedirs('../output', exist_ok=True)

# ================= 2. 数据加载与预处理 =================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 注意：如果你手动下载了，download=True 不会重复下载，而是直接加载本地文件
try:
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
except Exception as e:
    print(f"数据加载失败，请检查 ../data/MNIST/raw/ 目录下是否有4个 .gz 文件。错误信息: {e}")
    exit()

# 划分验证集（从训练集中取 20%）
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

# ================= 3. 定义 MLP 模型 =================
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ================= 4. 训练与评估循环 =================
epochs = 15
train_losses, val_losses = [], []
train_accs, val_accs = [], []
test_accs = []  # 【新增】用于记录每个epoch的测试集准确率

best_val_acc = 0.0

print("\n开始训练...")
for epoch in range(epochs):
    # ---- 训练阶段 ----
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

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- 验证阶段 ----
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

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    # ---- 【新增】测试集评估阶段（每个Epoch跑一次） ----
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    epoch_test_acc = test_correct / test_total

    scheduler.step()

    # 记录数据
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(epoch_test_acc)  # 记录测试集准确率

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
          f"Test Acc: {epoch_test_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '../output/best_mlp_model.pth')

# ================= 5. 最终测试评估 =================
model.load_state_dict(torch.load('../output/best_mlp_model.pth'))
model.eval()
final_test_correct, final_test_total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        final_test_correct += (predicted == labels).sum().item()
        final_test_total += labels.size(0)

final_test_acc = final_test_correct / final_test_total
print(f"\n===== 最终测试集准确率: {final_test_acc:.4f} ({final_test_acc*100:.2f}%) =====")

# ================= 6. 绘图一：Loss 与 Accuracy 曲线 =================
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：Loss 曲线
axes[0].plot(range(1, epochs+1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, epochs+1), val_losses, 'r-o', label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：Accuracy 曲线（包含 Train, Val, Test）
axes[1].plot(range(1, epochs+1), train_accs, 'b-o', label='Train Acc')
axes[1].plot(range(1, epochs+1), val_accs, 'r-o', label='Val Acc')
axes[1].plot(range(1, epochs+1), test_accs, 'g-o', label='Test Acc')  # 测试集曲线
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/mnist_curves.png', dpi=150)
plt.close() # 关闭当前画板，避免和后面冲突
print("Loss 与 Accuracy 曲线已保存到 ../output/mnist_curves.png")

# ================= 7. 绘图二：可视化至少3张手写数字图片 =================
model.eval()
# 从测试集中取一个 batch 的数据
test_images, test_labels = next(iter(test_loader))
test_images, test_labels = test_images.to(device), test_labels.to(device)

with torch.no_grad():
    outputs = model(test_images[:6])
    _, preds = torch.max(outputs, 1)

fig2, axes2 = plt.subplots(1, 3, figsize=(10, 4))
for i in range(3):
    # 把 tensor 转换成 numpy，并去掉 channel 维度 (1, 28, 28) -> (28, 28)
    img = test_images[i].cpu().squeeze().numpy()
    pred = preds[i].item()
    true = test_labels[i].item()
    
    axes2[i].imshow(img, cmap='gray')
    axes2[i].set_title(f'Pred: {pred} | True: {true}', 
                       color='green' if pred == true else 'red')
    axes2[i].axis('off')

plt.tight_layout()
plt.savefig('../output/mnist_visualization.png', dpi=150)
plt.close()
print("可视化预测图片已保存到 ../output/mnist_visualization.png")