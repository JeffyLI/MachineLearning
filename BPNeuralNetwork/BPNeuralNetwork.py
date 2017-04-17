from numpy import *

def sigmoid(z):
    y=[]
    for i in z:
        y.append(1/(1+exp(-i)))
    return y

def trainNetwork(smaple,label):
    sample_num = len(sample)
    sample_len = len(sample[0])
    out_num = 10
    hid_num = 8
    w1 = 0.2 * random.random((sample_len, hid_num)) - 0.1
    w2 = 0.2 * random.random((hid_num, out_num)) - 0.1
    hid_offset = zeros(hid_num)
    out_offset = zeros(out_num)
    input_learnrate = 0.2
    hid_learnrate = 0.2
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
        out_delta=err*out_act*(1-out_act) #输出层的方向梯度方向                         
        hid_delta = hid_act*(1 - hid_act) * dot(w2, out_delta)   
        for j in range(0,out_num):
            w2[:,j]+=hid_learnrate*out_delta[j]*hid_act
        for k in range(0,hid_num):
            w1[:,k]+=input_learnrate*hid_delta[k]*sample[i]

        out_offset += hid_learnrate * out_delta   #阈值的更新                    
        hid_offset += input_learnrate * hid_delta

    return w1,w2,hid_offset,out_offset

input_x=[-5,-1,0,1,5]
print(sigmoid(input_x))
