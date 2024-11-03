import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

# 随机生成800个点
num_points = 800
x = np.random.uniform(-1, 1, num_points)
y = np.random.uniform(-1, 1, num_points)
z = 1 * np.exp(-(x**2 + y**2) / 0.6**2)

# 创建Streamlit界面
st.title("3D 点与可调 Z 平面")

# 添加滑块来调整Z值
z_value = st.slider("调整XY平面Z值", min_value=0.1, max_value=0.9, value=0.5)

# 根据z_value判断点的颜色
colors = np.where(z > z_value, 'red', 'blue')

# 准备数据用于SVM
X = np.vstack((x, y)).T
y_labels = np.where(z > z_value, 1, 0)  # 1表示上方，0表示下方

# 训练SVM模型
model = svm.SVC(kernel='linear')
model.fit(X, y_labels)

# 生成网格以预测类别
xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z_svm = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点
ax.scatter(x, y, z, c=colors, marker='o')

# 绘制XY平面
zz = np.full(xx.shape, z_value)
ax.plot_surface(xx, yy, zz, alpha=0.5, color='gray')

# 绘制SVM决策边界
ax.contourf(xx, yy, Z_svm, alpha=0.3, cmap='coolwarm', zdir='z', offset=0)

# 设置轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 显示图形
st.pyplot(fig)
