import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========== 1. 设置真实参数 ==========
# y = a1*x1^2 + a2*x1*x2 + a3*sin(x1) + a4*cos(2*x2) + exp(a5*x1) + a6*x2 + b
a1_true, a2_true, a3_true = 2.0, 1.5, 3.0
a4_true, a5_true, a6_true = 0.5, -0.5, 0.8
b_true = 2026.0

# ========== 2. 生成数据 ==========
torch.manual_seed(42)
np.random.seed(42)

# 生成 x1, x2 的网格数据 (范围为 -3 到 3)
x1 = torch.linspace(-3, 3, 50)
x2 = torch.linspace(-3, 3, 50)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

# 计算真实的无噪声 y 值
Y_true = (a1_true * X1**2 + a2_true * X1 * X2 + a3_true * torch.sin(X1) +
          a4_true * torch.cos(2 * X2) + torch.exp(a5_true * X1) +
          a6_true * X2 + b_true)

# 加入噪声（-3 到 3 之间的随机数）
noise = (torch.rand_like(Y_true) * 6) - 3  # [0,1)映射到[-3, 3)
Y_noisy = Y_true + noise

# 展平数据用于训练
x1_flat = X1.flatten().reshape(-1, 1)
x2_flat = X2.flatten().reshape(-1, 1)
Y_target = Y_noisy.flatten().reshape(-1, 1)

# ========== 3. 定义待拟合参数 ==========
# 使用 PyTorch 的 Parameter 包装，使其可求导
a1 = torch.nn.Parameter(torch.randn(1))
a2 = torch.nn.Parameter(torch.randn(1))
a3 = torch.nn.Parameter(torch.randn(1))
a4 = torch.nn.Parameter(torch.randn(1))
a5 = torch.nn.Parameter(torch.randn(1) * 0.1) # 指数项参数要小一点，防止梯度爆炸
a6 = torch.nn.Parameter(torch.randn(1))
b  = torch.nn.Parameter(torch.randn(1) + 2026) # 让 b 初始值接近真实值

# 收集参数列表
params = [a1, a2, a3, a4, a5, a6, b]

# 优化器
optimizer = torch.optim.Adam(params, lr=0.01)
epochs = 5000

# ========== 4. 训练循环 ==========
losses = []
print("开始训练...")

for epoch in range(epochs):
    # 前向传播（使用待拟合公式）
    Y_pred = (a1 * x1_flat**2 + a2 * x1_flat * x2_flat + a3 * torch.sin(x1_flat) +
              a4 * torch.cos(2 * x2_flat) + torch.exp(a5 * x1_flat) +
              a6 * x2_flat + b)
    
    # 计算 MSE 损失
    loss = torch.mean((Y_pred - Y_target)**2)
    
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.2f}")

# ========== 5. 计算 R² ==========
with torch.no_grad():
    Y_final_pred = (a1 * x1_flat**2 + a2 * x1_flat * x2_flat + a3 * torch.sin(x1_flat) +
                    a4 * torch.cos(2 * x2_flat) + torch.exp(a5 * x1_flat) +
                    a6 * x2_flat + b)
    ss_res = torch.sum((Y_target - Y_final_pred)**2)
    ss_tot = torch.sum((Y_target - torch.mean(Y_target))**2)
    r2 = 1 - ss_res / ss_tot

print(f"\n===== 拟合结果 =====")
print(f"真实参数: a1={a1_true}, a2={a2_true}, a3={a3_true}, a4={a4_true}, a5={a5_true}, a6={a6_true}, b={b_true}")
print(f"拟合参数: a1={a1.item():.3f}, a2={a2.item():.3f}, a3={a3.item():.3f}, a4={a4.item():.3f}, a5={a5.item():.3f}, a6={a6.item():.3f}, b={b.item():.3f}")
print(f"R² = {r2.item():.6f}")

# ========== 6. 绘制三维图像 ==========
# 准备绘图网格
Y_true_grid = Y_true.detach().numpy()
Y_fit_grid = Y_final_pred.reshape(X1.shape).detach().numpy()

fig = plt.figure(figsize=(14, 6))

# 图1：真实函数（不含噪声）+ 采样点
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1.numpy(), X2.numpy(), Y_true_grid, cmap='viridis', alpha=0.8, edgecolor='none')
# 画出带噪声的采样点
ax1.scatter(X1.flatten()[::50], X2.flatten()[::50], Y_noisy.flatten()[::50], c='red', s=5, label='Sampled Points')
ax1.set_title('Ground Truth Function (R² = 0.9486)', fontsize=12)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.legend()

# 图2：拟合函数
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1.numpy(), X2.numpy(), Y_fit_grid, cmap='plasma', alpha=0.8, edgecolor='none')
ax2.set_title(f'Fitted Function (R² = {r2.item():.4f})', fontsize=12)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

plt.tight_layout()
plt.savefig('../output/least_squares_3d_result.png', dpi=150)
plt.show()
print("三维图像已保存到 ../output/least_squares_3d_result.png")