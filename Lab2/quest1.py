import numpy as np

X=[[1,0,2],[0,1,1],[2,1,0],[1,1,1],[0,2,1]]
row=len(X)
col=len(X[0])
s=[0]*col
for i in range(row):
    for j in range(col):
        s[j]=s[j]+X[i][j]
avg=[s[j]/row for j in range(col)]
print(avg)

#create mean centered matrix

Xc=[[0]*col for _ in range(row)]
for i in range(row):
    for j in range(col):
        Xc[i][j]=(X[i][j]-avg[j])
print(Xc)

#create the transpose

rows=len(Xc)
cols=len(Xc[0])
XcT=[]
for i in range(cols):
    r=[]
    for j in range(rows):
        r.append(Xc[j][i])
    XcT.append(r)
print(XcT)

#matrix multiplication

r1=len(XcT)
c1=len(XcT[0])
c2=len(Xc[0])
res=[[0]*c2 for _ in range(r1)]
for i in range(r1):
    for j in range(c2):
        for k in range(c1):
              res[i][j]=res[i][j]+Xc[k][j]*XcT[i][k]
print(res)

#calculate covariance
cov=[[0]*c2 for _ in range(r1)]
for i in range(r1):
    for j in range(c2):
        cov[i][j]=res[i][j]/4
print(cov)



x=np.array([[1,0,2],[0,1,1],[2,1,0],[1,1,1],[0,2,1]])
print(x)
cov=np.cov(x,rowvar=False)
print(cov)