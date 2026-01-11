import matplotlib.pyplot as plt

start=-10
stop=10
n=100
step=(stop-start)/(n-1)
x=[]
for i in range(n):
    x.append(start+i*step)
print(x)
y=[(j**2) for j in x]
plt.plot(x,y)
d=[-5,-3,0,3,5]
dy=[2*i for i in d]
print("derivatives=", dy)
plt.plot(d,dy)
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.show()




