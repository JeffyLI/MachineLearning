import numpy
import operator

'''
计算一个点和其他点的距离：
point   一个点
dataset 其他点
'''
def m_distance(point,dataset):
    result=[]
    for i in range(len(dataset)):
        t=0
        for j in range(len(point)):
            t+=(point[j]-dataset[i][j])**2
        result.append(t**0.5)
    return result

'''
分类器：
point   新加入的点
dataset 样本点集合
label   样本点对应标签
k       邻居个数
'''
def classify(point,dataset,label,k):
    dis=m_distance(point,dataset)
    sortIndex=numpy.argsort(dis)
    result={}
    for i in range(k):
        c=label[sortIndex[i]]
        result[c]=result.get(c,0)+1
    sortresult=sorted(result.items(),key=operator.itemgetter(1),reverse=True)
    return sortresult[0][0]


a=[5,5]
b=[[1,1],[0,2],[5,4],[3,4]]
label=[0,0,1,1]
print(classify(a,b,label,2))
