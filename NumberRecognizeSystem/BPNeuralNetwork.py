from numpy import *

def getDataFromTxt(filename):
    dataArr=[]
    file=open(filename)
    for line in file.readlines():
        for i in range(len(line)-1):
            dataArr.append(int(line[i]))
    return dataArr

def sigmoid(z):
    y=[]
    for i in z:
        y.append(1/(1+exp(-i)))
    return y

def trainNetwork(sample,label):
    sample_num = len(sample)
    sample_len = len(sample[0])
    out_num = 10
    hid_num = 20
    w1 = 0.2 * random.random((sample_len, hid_num)) - 0.1
    w2 = 0.2 * random.random((hid_num, out_num)) - 0.1
    hid_offset = zeros(hid_num)
    out_offset = zeros(out_num)
    input_learnrate = 0.3
    hid_learnrate = 0.3
    for i in range(0,len(sample)):
        t_label=zeros(out_num)
        t_label[label[i]]=1
        #前向的过程
        hid_value=dot(sample[i],w1)+hid_offset #隐层的输入            
        hid_act=sigmoid(hid_value)                 #隐层对应的输出                                 
        out_value=dot(hid_act,w2)+out_offset
        out_act=sigmoid(out_value)    #输出层最后的输出                                 

        #后向过程
        err=t_label-out_act
        out_delta=err*out_act #输出层的方向梯度方向                         
        hid_delta = hid_act*add(-1*ones(len(hid_act)), hid_act) * dot(w2, out_delta)   
        for j in range(0,out_num):
            w2[:,j]=add(w2[:,j],multiply(hid_learnrate*out_delta[j],hid_act))
        for k in range(0,hid_num):
            w1[:,k]=add(w1[:,k],multiply(input_learnrate*hid_delta[k],sample[i]))

    return w1,w2,hid_offset,out_offset

def testBPNN(sample,testlabel,w1,w2,hid_offset,out_offset):
    result=0
    for i in range(0,len(sample)):
        hid_value=dot(sample[i],w1)+hid_offset #隐层的输入            
        hid_act=sigmoid(hid_value)                 #隐层对应的输出                                 
        out_value=dot(hid_act,w2)+out_offset
        out_act=sigmoid(out_value)    #输出层最后的输出
        print(out_act)
        print(argmax(out_act))
        if argmax(out_act)==testlabel[i]:
            result+=1
    return float(result)/len(sample)

dataMatrix=[]
label=[]
for j in range(50):
    filename='trainingDigits\\3_'
    nfilename=filename+str(j)
    nfilename+='.txt'
    dataMatrix.append(getDataFromTxt(nfilename))
    label.append(1)
w1,w2,hid_offset,out_offset=trainNetwork(dataMatrix,label)
nfilename='trainingDigits\\3_2.txt'
testdata=[]
testlabel=[]
testdata.append(getDataFromTxt(nfilename))
testlabel.append([3])
testBPNN(testdata,testlabel,w1,w2,hid_offset,out_offset)

'''
dataMatrix=[]
label=[]
for j in range(100):
    for i in range(3):
        filename='trainingDigits\\'
        filename+=str(i)
        filename+='_'
        nfilename=filename+str(j)
        nfilename+='.txt'
        dataMatrix.append(getDataFromTxt(nfilename))
        label.append(i)      
w1,w2,hid_offset,out_offset=trainNetwork(dataMatrix,label)
nfilename='trainingDigits\\0_2.txt'
testdata=[]
testlabel=[]
testdata.append(getDataFromTxt(nfilename))
testlabel.append([1])
testBPNN(testdata,testlabel,w1,w2,hid_offset,out_offset)
'''
