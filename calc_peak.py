import numpy as np
import scipy.optimize as opt

def height_function(xs):
    beta = 0.2
    max_x = 10
    decay = 1 - beta * (xs / max_x)
    decay = np.clip(decay, 0.8, 1.0)
    return np.sin(2.2 * xs) * (0.6 + 0.3 * (xs / 1.8)) * decay + 0.3 * (xs + 1.2) + 2.5

# 用于寻找极大值的辅助函数
def find_max(func, lower_bound, upper_bound):
    # 通过最小化负值来最大化目标函数
    result = opt.minimize_scalar(lambda x: -func(x), bounds=(lower_bound, upper_bound), method='bounded')
    return result.x  # 返回找到的极大值的 x 坐标

# 查找区间内的极大值
peaks = []
for x_start in np.linspace(0, 3.8, 10):  # 在区间内分成多个点来查找每个波峰
    peak = find_max(height_function, x_start - 1, x_start + 1)  # 在每个区间附近查找极大值
    if peak not in peaks:  # 避免重复的波峰
        peaks.append(peak)

print("波峰位置:", peaks)