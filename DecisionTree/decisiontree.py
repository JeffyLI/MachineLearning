from math import log
import operator

'''
计算给定数据集的熵
dataset 数据集，最后一列应为类别
'''
def calcShannonEnt(dataset):
    numEntries=len(dataset)
    labelcount={}
    for featVec in dataset:
        currentLabel =featVec[-1]
        if currentLabel not in labelcount.keys():
            labelcount[currentLabel]=0
        labelcount[currentLabel]+=1
    shannonEnt=0.0
    for key in labelcount:
        p=float(labelcount[key])/numEntries
        shannonEnt-=p*log(p,2)
    return shannonEnt

'''
根据给定特征值分类数据集
dataset 待分类数据集
axis    给定特征值的下标
value   需要返回分类的特征值
'''
def splitDataSet(dataset,axis,value):
    retDataset=[]
    for featVec in dataset:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataset.append(reducedFeatVec)
    return retDataset

'''
选择最好的数据集分类
data 待分类数据集
'''
def chooseBestFeaturetoSplit(dataset):
    numFeature=len(dataset[0])-1
    baseShannonEnt=calcShannonEnt(dataset)
    bestInfo=0.0
    bestFeature=-1;
    for i in range(numFeature):
        featList=[example[i] for example in dataset]
        uniqueVals=set(featList)
        newShannonEnt=0.0
        for value in uniqueVals:
            subDataset=splitDataSet(dataset,i,value)
            p=len(subDataset)/float(len(dataset))
            newShannonEnt+=p*calcShannonEnt(subDataset)
        info=baseShannonEnt-newShannonEnt
        if info>bestInfo:
            bestInfo=info
            bestFeature=i
    return bestFeature

'''
采用多数表决决定分类
classlist  分类的类别
'''
def majorityCnt(classlist):
    classcount={}
    for vote in classlist:
        if vote not in classcount:
            classcount[vote]=0
        classcount[vote]+=1
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

'''
构建决策树
dataset 数据集
labels  特征标签
'''
def createDecisionTree(dataset,reallabels):
    labels=reallabels[:]
    classlist=[example[-1] for example in dataset]
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    if len(dataset[0])==1:
        return majorityCnt(classlist)
    bestFeature=chooseBestFeaturetoSplit(dataset)
    bestlabel=labels[bestFeature]
    myTree={bestlabel:{}}
    del(labels[bestFeature])
    featValues=[example[bestFeature] for example in dataset]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        sublabels=labels[:]
        myTree[bestlabel][value]=createDecisionTree(splitDataSet(dataset,bestFeature,value),sublabels)
    return myTree

'''
用决策树进行分类
inputTree   决策树
featlabels  特征值标签
testVec     测试特征值
'''
def classify(inputTree,featlabels,testVec):
    firststr=list(inputTree.keys())[0]
    secondDict=inputTree[firststr]
    featIndex=featlabels.index(firststr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]) is dict:
                classlabel=classify(secondDict[key],featlabels,testVec)
            else:
                classlabel=secondDict[key]
    return classlabel
    

dataset=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
labels=['no surfacing','flippers']
print(classify(createDecisionTree(dataset,labels),labels,[0,0]))
print(classify(createDecisionTree(dataset,labels),labels,[1,1]))
        
