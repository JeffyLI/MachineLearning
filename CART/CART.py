from numpy import *


'''
回归树
'''
'''
把数据集根据给定特征分成两类
dataset  数据集
feature  划分的特征
value    划分阀值
'''
def SplitDataset(dataset,feature,value):
    mat0=dataset[nonzero(dataset[:,feature]>value)[0],:]
    mat1=dataset[nonzero(dataset[:,feature]<=value)[0],:]
    return mat0,mat1

#求dataset最后一列的平均数
def regLeaf(dataset):
    return mean(dataset[:,-1])

#求dataset最后一列所有数据的方差
def regErr(dataset):
    return var(dataset[:,-1])*shape(dataset)[0]

'''
选择划分dataset的最优方案
dataset   要划分的数据集
leafType  求dataset最后一列的平均数
errTpye   求dataset最后一列所有数据的方差
ops       (方差进步最低值，子集元素最少个数)
'''
def chooseBestSplit(dataset,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]
    tolN=ops[1]
    if len(set(dataset[:,-1].T.tolist()))==1:
        return None,leafType(dataset)
    m,n=shape(dataset)
    S=errType(dataset)
    bestS=inf
    bestIndex=0
    bestValue=0
    #求取各种分类方案中方差最小的方案
    for featIndex in range(n-1):
        for splitVal in set(dataset[:,featIndex]):
            mat0,mat1=SplitDataset(dataset,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    #确保分类后的方差足够小
    if (S-bestS)<tolS:
        return None,leafType(dataset)
    return bestIndex,bestValue


'''
构建分类回归树
dataset   要划分的数据集
leafType  求dataset最后一列的平均数
errTpye   求dataset最后一列所有数据的方差
ops       (方差进步最低值，子集元素最少个数)
'''
def createTree(dataset,leafType=regLeaf,errType=regErr,ops=(0.1,4)):
    feat,val=chooseBestSplit(dataset,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet=SplitDataset(dataset,feat,val)
    retTree['left']=createTree(lSet,leafType,errType,ops)
    retTree['right']=createTree(rSet,leafType,errType,ops)
    return retTree

#判断输入是否为树
def isTree(obj):
    return (type(obj) is dict)

#递归遍历树，返回树的均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right']=getMean(tree['right'])
    if isTree(tree['left']):
        tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

'''
后剪枝函数
tree      待剪枝的树
testData  剪枝需要的测试数据集
'''
def prune(tree,testData):
    if shape(testData)[0]==0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lset,rset=SplitDataset(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lset)
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rset)
    if not isTree(tree['right']) and not isTree(tree['left']):
        lset,rset=SplitDataset(testData,tree['spInd'],tree['spVal'])
        errorNoMerge=sum(power(lset[:,-1]-tree['left'],2))+\
                    sum(power(rset[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2.0
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree



'''
模型树
用modeLeaf和modeErr代替createTree相应的参数即可构建模型树
'''
def linearSolve(dataset):
    m,n=shape(dataset)
    X=mat(ones((m,n)))
    Y=mat(ones((m,1)))
    X[:,1:n]=dataset[:,0:n-1];
    Y=mat(dataset[:,-1]).T
    xTx=X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular')
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

#相当于回归树的regLeaf
def modeLeaf(dataset):
    ws,X,Y=linearSolve(dataset)
    return ws

#相当于回归树的regErr
def modelErr(dataset):
    ws,X,Y=linearSolve(dataset)
    yHat=X*ws
    return sum(power(Y-yHat,2))

'''
用树回归进行预测的代码
'''
def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n=shape(inDat)[1]
    X=mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[0,tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m=len(testData)
    yHat=mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree,testData[i,:],modelEval)
    return yHat

#测试回归树
def testRegTree(testMat):
    mytreereg=createTree(testMat,regLeaf,regErr,(0.01,2))
    testMat=mat(testMat)
    yHatreg=createForeCast(mytreereg,testMat[:,:3])
    print(corrcoef(yHatreg,testMat[:,-1],rowvar=0)[0,1])

#测试模型树
def testModelTree(testMat):
    mytreemodel=createTree(testMat,modeLeaf,modelErr,(0.01,2))
    testMat=mat(testMat)
    yHatmodel=createForeCast(mytreemodel,testMat[:,:3],modelTreeEval)
    print(corrcoef(yHatmodel,testMat[:,-1],rowvar=0)[0,1])
        
testMat=random.rand(16,4)
regMat=testMat.copy()
modelMat=testMat.copy()
testRegTree(regMat)
testModelTree(modelMat)

