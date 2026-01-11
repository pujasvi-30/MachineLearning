import matplotlib.pyplot as plt
import math

mean=0
dev=15
var=dev**2
start=-100
stop=100
n=100
step=(stop-start)/(n-1)
x=[]
for i in range(n):
    x.append(start+i*step)
print(x)
y=[]
for j in x:
    a=(1/(dev*math.sqrt(2*3.14)))
    b=((j-mean)**2)/(2*var)
    y.append(a*math.exp(-b))
print(y)
plt.plot(x,y)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Guassian PDF")
plt.show()
