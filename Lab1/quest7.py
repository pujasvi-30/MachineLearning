t=[2,3,3]
X=[
    [1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]
]
r=len(X)
c=len(X[0])
res=[]
for i in range(r):
    sum=0
    for j in range(c):
        sum+=X[i][j]*t[j]
    res.append(sum)
print(res)

