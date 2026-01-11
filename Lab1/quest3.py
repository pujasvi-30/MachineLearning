import matplotlib.pyplot as plt

start= -10
stop=10
n=100
step=(stop-start)/(n-1)
x=[]
for i in range(n):
    x.append(round(start+i*step))
print(x)
y=[(round(2*(j**2))+(3*j)+4) for j in x]
plt.plot(x,y)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()