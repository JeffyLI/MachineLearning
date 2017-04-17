from numpy import *

'''
标准回归函数
xArr   特征值数据集
yArr   结果值数据集
返回值 斜率
'''
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

'''
局部加权线性回归函数
testPoint  测试点
xArr       x数据集
yArr       y数据集
k          控制衰减速率
返回值     testpoint的y值
'''
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye(m))
    for j in range(m):
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

'''
当数据特征数比样本还要多的时，矩阵就不是满秩矩阵，求逆会出现问题。
解决方案：岭回归（ridge regression），lasso，向前逐步回归
由于lasso计算复杂，而且用向前逐步回归也能得到类似效果，所以lasso算法就略过
'''

'''
岭回归
'''
def ridgeRegression(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    #标准化数据
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMean=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMean)/xVar
    #在30个不同的lam中调用ridgeRegres
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegression(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat


'''
向前逐步回归
'''
'''
标准化输入的矩阵
'''
def regularize(xMat):
    inMat = xMat.copy()  
    inMeans = mean(inMat,0) 
    inVar = var(inMat,0) 
    inMat = (inMat - inMeans)/inVar  
    return inMat

'''
向前逐步回归函数
xArr   x值集
yArr   y值集
eps    步长
numIt  迭代次数
返回   各个特征的权值
'''
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1))
    wsTest=ws.copy()
    
    wsMax=ws.copy()
    for i in range(numIt):
        lowestError=inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=sum(multiply(yMat.A-yTest.A,yMat.A-yTest.A))
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

dataset=[[1.0,0.3],[2.0,0.5],[3.0,0.9],
         [4.0,1.0],[5.0,1.3],[6.0,1.1],
         [7.0,1.6],[8.0,2.5]]
yArr=[1.0,1.5,2.5,2.8,4.0,3.8,6.0,5.3]

'''
print(stageWise(dataset,yArr,0.2,30))
'''
print(ridgeTest(dataset,yArr))
print(standRegres(dataset,yArr))


'''
result=zeros(len(yArr))
for i in range(len(yArr)):
    result[i]=lwlr(dataset[i],dataset,yArr,0.5)
print(result)
print(lwlr([9.0],dataset,yArr,0.5))
'''
