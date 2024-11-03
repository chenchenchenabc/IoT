import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

# 隨機生成點
num_points = 800
x = np.random.uniform(-1, 1, num_points)
y = np.random.uniform(-1, 1, num_points)
z = 1 * np.exp(-(x**2 + y**2) / 0.6**2)

# 調整平面 z 值
z_plane = st.slider('Select Z value for XY plane', 0.1, 0.9, 0.5)

# SVM 訓練
points = np.vstack((x, y)).T
labels = (z > z_plane).astype(int)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(points, labels)
z_svm = clf.decision_function(points).reshape(x.shape)

# 調整斜率
x_slope = st.slider('Select X slope', -2.0, 2.0, 0.0)
y_slope = st.slider('Select Y slope', -2.0, 2.0, 0.0)

# 調整 z 平面
z_plane_adjusted = z_plane + x_slope * x + y_slope * y

# 計算顏色
colors = np.where(z > z_plane_adjusted, 'red', 'blue')

# 繪製 3D 圖
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors, label='Data Points')

# 創建網格以繪製平面
X_plane, Y_plane = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
Z_plane = z_plane + x_slope * X_plane + y_slope * Y_plane

# 繪製平面
ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.5, color='gray')

# 設置 z 軸範圍
ax.set_zlim(0, 1.5)  # 調整此範圍以顯示所有點

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Scatter Plot with SVM Decision Boundary and XY Plane')
st.pyplot(fig)
