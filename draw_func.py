import numpy as np
import matplotlib.pyplot as plt

def modified_height(xs):

    y = 0.8* np.sin(0.5*xs) + 0.08*xs
    return y

# 生成数据
xs = np.linspace(-9, 6, 500)

# 调用修改后的函数
ys = modified_height(xs)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(xs, ys, label="Modified Height Function")
plt.title("Function with Increasing Peak Height in the Front")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
