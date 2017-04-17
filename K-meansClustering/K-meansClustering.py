from numpy import *

#求向量AB的距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#随机产生k个簇质心
def randCent(dataset,k):
    n=shape(dataset)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataset[:,j])
        rangeJ=float(max(dataset[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return centroids

'''
K均值聚类算法
dataset   待分类的数据
k         类别数
'''
def kMeans(dataset,k):
    m=shape(dataset)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=randCent(dataset,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        #计算每个点最近的簇质点
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(k):
                distJI=distEclud(centroids[j,:],dataset[i,:])
                if distJI<minDist:
                    minDist=distJI;
                    minIndex=j
            if clusterAssment[i,0] != minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        #更新簇质心
        for cent in range(k):
            pstClust=dataset[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(pstClust,axis=0)
    return centroids,clusterAssment


'''
二分K均值聚类算法
dataset   待分类的数据
k         类别数
'''
def biKmeans(dataset,k):
    m=shape(dataset)[0]
    clusterAssment=mat(zeros((m,2)))
    centroid0=mean(dataset,axis=0).tolist()[0]
    centlist=[centroid0]
    for j in range(m):
        clusterAssment[j,1]=distEclud(mat(centroid0),dataset[j,:])**2
    while (len(centlist)<k):
        lowestSSE=inf
        #尝试所有方案，求最能降低SSE的划分方案
        for i in range(len(centlist)):
            ptsIncurrCluster=dataset[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss=kMeans(ptsIncurrCluster,2)
            sseSplit=sum(splitClustAss[:,1])
            sseNotSplit=sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            if (sseSplit+sseNotSplit)<lowestSSE:
                bestCentToSplit=i
                bestNewCents=centroidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNotSplit
        #更新质子表和聚类表
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0]=len(centlist)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        centlist[bestCentToSplit]=bestNewCents[0,:]
        centlist.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return centlist,clusterAssment    
    
    

testMat=mat(random.rand(8,4)*12)
mylist,mycluster=biKmeans(testMat,3)
print(testMat)
print(mycluster)
