"""两个节点共用的几何小工具。

现在只有一个函数，但它被 `fake_camera_node`（发真值）和 `fuse_node`（发检测结果）
同时用到。复制两份的话，哪天改了其中一份而忘了另一份，RViz 里真值和检测结果的
朝向就会对不上，而那种偏差肉眼很难分辨。

不引 tf_transformations：那个包不一定装，而这里只需要「让局部 z 轴对准某方向」，
二十行就够。
"""

import numpy as np


def quaternion_from_z_axis(normal) -> tuple:
    """构造一个让局部 z 轴对准给定方向的四元数，返回 (x, y, z, w)。

    为什么需要它
        RViz 显示箭头、控制组算预吸附位，都需要一个姿态而不是一个裸向量。
        四元数的约定是「把局部坐标系的 z 轴转到这个方向」，所以给方向反推四元数
        是这一步的标准做法。

    用 Shepperd 分支写法
        常见的 `w = sqrt(1+trace)/2` 单支写法在 trace 接近 -1 时分母趋零，
        数值不稳。四个分支分别以最大的对角元为分母，避免这个问题。
    """
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    n = n / norm

    # 任取一个与 n 不平行的向量构造正交基
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, n)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    m = np.stack([x_axis, y_axis, n], axis=1)

    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)
