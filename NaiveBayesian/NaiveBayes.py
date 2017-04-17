from numpy import *
import re

'''
获取单词表
fileName 文件名
'''
def GetData(fileName):
    file=open(fileName)
    string=file.read()
    file.close()
    rawWords=re.split('\W+',string)
    listofwords=[tok.lower() for tok in rawWords if len(tok)>2]
    return listofwords

'''
创建单词表：dataset里面的单词合并成一个集合（无重复元素）
'''
def createVocabList(dataset):
    vocabset=set([])
    for document in dataset:
        vocabset=vocabset|set(document)
    return list(vocabset)


'''
把文档中的单词转换成特征向量
vocabList 单词表
inputset  输入的单词或文档
'''
def setofwordsVec(vocabList,inputset):
    returnVec=[0]*len(vocabList)
    for word in inputset:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1;
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec

'''
训练朴素贝叶斯
trainMatrix   训练的特征向量集
trainCategory 特征向量集对应的标签分类集
'''
def trainNB(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #初始化改成1和2.0是为了用对数求结果
    p0num=ones(numWords)
    p0Denom=2.0
    p1num=ones(numWords)
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    #用对数求结果为了避免下溢出
    p1vect=log(p1num/p1Denom)
    p0vect=log(p0num/p0Denom)
    return p0vect,p1vect,pAbusive

'''
根据概率大小输出结果
'''
def classifyNB(vecClassify,p0Vec,p1Vec,pClass1):
    p1=sum(vecClassify*p1Vec)+log(pClass1)
    p0=sum(vecClassify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
'''
对朴素贝叶斯功能封装
vocabList   单词表
classVec    分类向量
testEntry   待分类的单词表
输出 分类结果
'''
def myNB(vocabList,classVec,testEntry):
    myVocabList=createVocabList(vocabList)
    trainMat=[]
    for postinDoc in vocabList:
        trainMat.append(setofwordsVec(myVocabList,postinDoc))
    a,b,c=trainNB(trainMat,classVec)
    thisDoc=setofwordsVec(myVocabList,testEntry)
    print(testEntry,'classified as: ',classifyNB(thisDoc,a,b,c))
    
dataset=[['my','dog','has','flea','problems','help','please'],
         ['maybe','not','take','him','to','dog','park','stupid'],
         ['my','dalmation','is','so','cute','I','love','him'],
         ['stop','posting','stupid','worthless','garbage'],
         ['mr','licks','ate','my','steak','how','to','stop','him'],
         ['quit','buying','worthless','dog','food','stupid']
         ]
classVec=[0,1,0,1,0,1]
testEntry=['maybe','not','take']
myNB(dataset,classVec,testEntry)

