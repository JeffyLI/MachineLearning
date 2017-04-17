from numpy import *
from operator import *
import math

'''
根据给定的阀值返回数据的分类结果
datMat       训练数据集（矩阵）
dimen        决定分类的特征
threshVal    决定分类的阀值
threshIneq   分类类别
'''
def stumpClassify(datMat,dimen,threshVal,threshIneq):
    retArray=ones((shape(datMat)[0],1))
    if threshIneq=='lt':
        retArray[datMat[:,dimen]<=threshVal]=-1.0
    else:
        retArray[datMat[:,dimen]>threshVal]=1.0
    return retArray

'''
获取数据集的最优弱分类器（单层决策树）
dataArr       训练数据集
classLabels   分类标签集
D             特征权重向量
'''
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClasEst=mat(zeros((m,1)))
    minError=inf
    for i in range(n):
        # 计算步长，即阀值每次循环的增加量
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst


'''
训练单层决策树
dataMat     训练的数据集
clssLabels  训练数据集的类别向量
numIt       迭代次数，默认40次
'''
def adaBoostTrainDS(dataMat,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataMat)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataMat,classLabels,D)
        alpha=float(0.5*log(math.e,(1.0-error)/max(error,0.0000001)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        #用expon的每个元素正负判断classLabels和classEst的相似程度
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        #aggClassEst与classLabels符号不同的元素挑选出来，计算错误率
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        if errorRate==0.0: break
    return weakClassArr


'''
分类给定数据集
dataToClass    待分类的数据集
classifierArr  训练后的单层决策树
'''
def adaClassify(dataToClass,classifierArr):
    dataMatrix=mat(dataToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],
                               classifierArr[i]['thresh'],
                               classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
    return sign(aggClassEst)

datMat=matrix([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.0],[2.0,1.0]])
classLabels=[1.0,1.0,-1.0,-1.0,1.0]
D=mat(ones((5,1))/5)
classifierArr=adaBoostTrainDS(datMat,classLabels,9)
print(adaClassify([2.0,1.0],classifierArr))
