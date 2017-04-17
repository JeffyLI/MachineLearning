from numpy import *

'''
构建类别集，选出给定数据集所有的类别
dataset  需要构建类别集的数据集
'''
def createClass(dataset):
    cl=[]
    for transaction in dataset:
        for item in transaction:
            if not [item] in cl:
                cl.append([item])
    cl.sort()
    return cl

'''
计算每个类别的支持度，并选出高于最低支持度的类别
dataset      数据集
ck           类别集
minSupport   最低的支持度
'''
def scanDataset(dataset,ck,minSupport):
    ssCnt={}
    for can in ck:
        can.append(0)
    for tid in dataset:
        for can in ck:
            if (set(can[:-1])&set(tid))==set(can[:-1]):
                can[-1]+=1
    numItems=float(len(dataset))
    retList=[]
    supportData={}
    for key in ck:
        support=key[-1]/numItems
        if support>minSupport:
            retList.append(key[0:-1])
        supportData[str(key[0:-1])]=support
    return retList,supportData

'''
计算给定数据集的所有的两两组合集
Lk   数据集
'''
def aprioriGen(Lk):
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk)
            Lin=list(set(L1[i])|set(L1[j]))
            if Lin not in retList:
                retList.append(Lin)
    return retList


'''
Apriori算法
dataset 数据集
minSupport 最低支持度（默认0.5）
'''
def apriori(dataset,minSupport=0.5):
    cl=createClass(dataset)
    L1,supportData=scanDataset(dataset,cl,minSupport)
    L=[L1]
    k=2
    while(len(L[k-2])>0):
        Ck=aprioriGen(L[k-2])
        Lk,supk=scanDataset(dataset,Ck,minSupport)
        supportData.update(supk)
        L.append(Lk)
        k+=1
    return L,supportData

'''
计算可信度
freqset      计算可信度需要的总集合
H            freqset的子集
supportData  支持度数据集
br1       保存符合要求的规则
minConf   最低可信度
'''
def calcConf(freqset,H,supportData,br1,minConf):
    prunedH=[]
    s=""
    for conseq in H:
        conf=supportData[str(freqset)]/supportData[str(list(set(freqset)-set(conseq)))]
        if conf>=minConf:
            s=str(set(freqset)-set(conseq))+'-->'+str(conseq)+'  conf: '+str(conf)
            br1.append(s)
            prunedH.append(conseq)
    return prunedH

#功能同上，用于freqest元素超过2个的情况
def rulesFromConseq(freqset,H,supportData,br1,minConf):
    m=len(H[0])
    if (len(freqset)>(m+1)):
        Hmp1=aprioriGen(H)
        Hmp1=calcConf(freqset,Hmp1,supportData,br1,minConf)
        if (len(Hmp1)>1):
            rulesFromConseq(freqset,Hmp1,supportData,br1,minConf)
'''
产生规则
L           符合要求的元素集  
supportData 支持度数据集
minConf     最低可信度
'''
def generateRules(L,supportData,minConf=0.7):
    bigRuleList=[]
    for i in range(1,len(L)-1):
        for freqset in L[i]:
            H1=[[item] for item in freqset]
            if i>1:
                rulesFromConseq(freqset,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqset,H1,supportData,bigRuleList,minConf)
    return bigRuleList
            

dataset=[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
L,supportData=apriori(dataset,0.4)
rules=generateRules(L,supportData,0.5)
for rule in rules:
    print(rule)
