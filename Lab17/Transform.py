import  numpy as np
import matplotlib.pyplot as plt
# X = np.array(
#     [
#         [1, 13],
#         [1, 18],
#         [2, 9],
#         [3, 6],
#         [6, 3],
#         [9, 2],
#         [13, 1],
#         [18,1],
#         [3, 15],
#         [6, 6],
#         [6, 11],
#         [9, 5],
#         [10, 10],
#         [11, 5],
#         [12,6],
#         [16, 3],
#     ]
# )
# y = np.array(
#     [
#         [0],
#         [0],
#         [0],
#         [0],
#         [0],
#         [0],
#         [0],
#         [0],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1]
#         ]
# )
# plt.figure(figsize=(5,4))
# plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", s=100, edgecolors="k")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Original 2D Data (Not Linearly Separable)")
# plt.show()

# def transform(X):
#     x1=X[:,0]
#     x2=X[:,1]
#     z1=x1**2
#     z2=np.sqrt(2)*x1*x2
#     z3=x2**2
#     return np.column_stack((z1,z2,z3))

def transform(x):
    x1,x2=x
    return(np.array([x1**2,x2**2,np.sqrt(2)*x1*x2]))
x1 = np.array([3, 6])
x2 =np.array([10, 10])
t1x = transform(x1)
t2x = transform(x2)
k=np.dot(t1x,t2x)
print("Using kernel:",k)
n=np.dot(x1,x2)
print("Normally:",n**2)


# X_transformed = transform(X)
# fig=plt.figure(figsize=(5,4))
# ax=fig.add_subplot(111,projection="3d")
# ax.scatter(X_transformed[:,0], X_transformed[:,1], X_transformed[:,2],c=y, cmap="coolwarm", s=100)
# ax.set_xlabel("x1^2")
# ax.set_ylabel("root2*x1*x2")
# ax.set_zlabel("x2^2")
# plt.show()

