A=[[1,2,3],[4,5,6]]
rows=len(A)
cols=len(A[0])
AT=[]
for i in range(cols):
    row=[]
    for j in range(rows):
        row.append(A[j][i])
    AT.append(row)
print(AT)




r=len(AT)
c=len(AT[0])
res=[[0 for m in range(c)] for n in range(rows)]
for i in range(rows):
    for j in range(c):
        for k in range(cols):
            res[i][j]=res[i][j]+A[i][k]*AT[k][j]

print(res)





#AT=[[0 for x in range(cols)] for y in range(cols)]
#for i in range(n):
#    for j in range(n):
#        for k in range(m):
#            AT[i][j]+=A[k][i]*A[k][j]
#print(AT)




