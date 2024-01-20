"""
逐元运算(element-wise)
条件：形状相同
"""
import numpy as np
x = np.array([[5,78,2,34,0],
              [6,79,3,35,1],
              [7,80,4,36,2]])
y = np.array([[6,79,3,35,1],
              [7,80,4,36,2],
              [5,78,2,34,0]])
# L0正则
def native_relu(x):
    assert len(x.shape) == 2    # 2D 张量
    x = x.copy()    # 避免覆盖输入张量
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

# 加法
def native_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x    

# 减法
def native_sub(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] -= y[i, j]
    return x    

# 乘法
def native_mul(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] *= y[i, j]
    return x    

print(native_relu(x))
print(native_add(x, y))

# 基础线性代数子程序（BLAS）
z1 = x + y
z2 = np.maximum(z1, 0.)
print(z1)
print(z2)

"""
广播
步骤：1.添加轴
      2.复制
"""
x = np.array([[5,78,2,34,0],
              [6,79,3,35,1],
              [7,80,4,36,2]])
y = np.array([12,3,6,14,7])
def native_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    print(x.shape[1])
    print(y.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x       
print(native_add_matrix_and_vector(x, y))

x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x,y)
print(z)

"""张量点积"""
# 向量与向量点积
def native_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

# 矩阵与向量点积
def native_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z      

def native_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = native_vector_dot(x[i,:],y)
    return z 

# 矩阵与矩阵点积
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros(x.shape[0],y.shape[1])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = native_matrix_vector_dot(row_x, column_y)
    return z

"""张量变形"""
x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)

x = x.reshape((6, 1))
print(x)
x = x.reshape((2, 3))
print(x)

# 转置
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)
x = x.T
print(x.shape)
