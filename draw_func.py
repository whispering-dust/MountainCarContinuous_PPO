import numpy as np
import matplotlib.pyplot as plt

def modified_height(xs, beta=0.1, max_x=10):
     # 衰减因子
    decay = 1 - beta * (xs / max_x)
    decay = np.clip(decay, 0.8, 1.0)  # 防止过度衰减，保持一定下限
    
    # 修改后的函数
    return np.sin(2.2 * xs) * (0.3 + 0.3 * (xs / 1.8)) * decay + 0.3 * (xs + 1.2) + 2.5

# 生成数据
xs = np.linspace(-6, 3.6, 500)

# 调用修改后的函数
ys = modified_height(xs, beta=0.2)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(xs, ys, label="Modified Height Function")
plt.title("Function with Increasing Peak Height in the Front")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
