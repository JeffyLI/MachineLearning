from numpy import *
import operator
from time import clock

'''
把文件的01数据提取到一个数组里面
filename 数据的文件名
'''
def getDataFromTxt(filename):
    dataArr=[]
    file=open(filename)
    for line in file.readlines():
        for i in range(len(line)-1):
            dataArr.append(int(line[i]))
    return dataArr

def m_distance(point,dataset):
    result=[]
    for i in range(len(dataset)):
        t=0
        for j in range(len(point)):
            t+=(point[j]-dataset[i][j])**2
        result.append(t**0.5)
    return result

def classify(point,dataset,label,k):
    dis=m_distance(point,dataset)
    sortIndex=argsort(dis)
    result={}
    for i in range(k):
        c=label[sortIndex[i]]
        result[c]=result.get(c,0)+1
    sortresult=sorted(result.items(),key=operator.itemgetter(1),reverse=True)
    return sortresult[0][0]

time1=clock()
dataMatrix=[]
label=[]
filename=''
for i in range(10):
    filename='trainingDigits\\'
    filename+=str(i)
    filename+='_'
    for j in range(30):
        nfilename=filename+str(j)
        nfilename+='.txt'
        dataMatrix.append(getDataFromTxt(nfilename))
        label.append(i)
        
time2=clock()
count=0
for i in range(10):
    filename='testDigits\\'
    filename+=str(i)
    filename+='_'
    for j in range(20):
        nfilename=filename+str(j)
        nfilename+='.txt'
        temp=classify(getDataFromTxt(nfilename),dataMatrix,label,5)
        if i==temp:
            count+=1
        else:
            print(nfilename,"error number: ",temp)
print("正确个数：",count)
print("正确率：",count/2.0,"%")
time3=clock()
print(time1," ",time2," ",time3)
print(time3-time2)

