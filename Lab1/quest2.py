
import matplotlib.pyplot as plt

# Simple data
start= -100
stop=100
n=100
step= (stop-start)/ n-1
x=[]
for i in range(n):
    x.append(start+i*step)
print(x)
y= [(2*j)+3 for j in x]
plt.plot(x,y)


#x = [1, 2, 3, 4]
#y = [1, 4, 9, 16]

# Create a plot
#plt.plot(x, y)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Matplotlib Test Plot")

# Show the plot
plt.show()

#print("Matplotlib is working correctly!")
