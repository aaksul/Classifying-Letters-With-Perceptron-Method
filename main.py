import My_NN
import numpy as np
import random
import matplotlib.pyplot as plt


Directory_Path=["Karakter_Tam","Karakter_Tam_binary"]
data_type=["bipolar","binary"]
perceptron_rules=["default","delta"]
input_type=-1
INPUT_LENGTH=63
INPUT_NUMBER=21
LABEL_NUMBER=7
def ReadFile(path,input_number,input_length,label_number,label_type):
    Letters=["A","B","C","D","E","J","K"]
    Fonts=["1","2","3"]
    labels=np.zeros(shape=(input_number,label_number))
    inputs=np.zeros(shape=(input_number,input_length))
    for x,font in enumerate(Fonts):
        for i,letter in enumerate(Letters):
            TxtFile="Font_"+font+"_"+letter+".txt"
            f = open(path+"/"+TxtFile,"r")
            string=f.read()
            string=string.replace("\n","")
            values=string.split(",")
            values=list(map(int,values))
            input=np.asarray(values)
            index=x*7+i
            inputs[index]=values
            if(label_type=="binary"):
                label=np.full(shape=7,fill_value=0)
            elif(label_type=="bipolar"):
                label=np.full(shape=7,fill_value=-1)
            label[i]=1
            labels[index]=label
    return inputs,labels

def Change_Pixel_Values(inputs,input_length,pixel_number,data_type):
    inps=inputs.copy()
    for inp in inps:
        for j in range(pixel_number):#
            index=j-1
            if data_type == "bipolar":
                inp[index]=inp[index]*(-1)#if value is -1 change to 1  if value is 1 , change to -1
            if data_type == "binary":
                inp[index]=float(1^int(inp[index]))#if value is 0 change to 1  if value is 1 , change to 0
    return inps

print("For default value just press Enter")
input_type=int(raw_input("Select the type of input (Enter \"0\" for bipolar,\"1\" for Binary)=>") or "0" )
learning_rate=float(raw_input("Enter learning_rate value( 0 < l <1 )[default=0.001]=>") or "0.001")
bias = float(raw_input("Enter the bias value[default=0]=> ") or "0")
threshold=float(raw_input("Enter the threshold value[default=1]=>") or "1")
epoch=int(raw_input("Enter epoch value[default=50]=>") or "50")
i=int(raw_input("Select perceptron rule[Enter \"0\" for default rule, \"1\" for delta rule ]=>") or "0")
perceptron_rule=perceptron_rules[i]




inputs,labels=ReadFile(Directory_Path[input_type],input_number=INPUT_NUMBER,input_length=INPUT_LENGTH,label_number=LABEL_NUMBER,label_type=data_type[input_type])
Model1=My_NN.Model(method="perceptron",input_type=data_type[input_type],perceptron_rule=perceptron_rule)
layer1=My_NN.layer(name="input_layer",type="input",nodes_number=INPUT_LENGTH)
layer2=My_NN.layer(name="label_layer",type="output",nodes_number=LABEL_NUMBER)
Model1.add_layer(layer=layer1)
Model1.add_layer(layer=layer2)
Model1.set_input_values(inputs,input_number=INPUT_NUMBER,input_length=INPUT_LENGTH)
Model1.set_output_values(labels,output_length=LABEL_NUMBER)
epoch_accuracy=Model1.train(learning_rate=learning_rate,epoch=epoch,bias=bias,threshold=threshold)


plt.xlim(0, len(epoch_accuracy)), plt.ylim(0, 1.1)
plt.xlabel('epoch count', fontsize=18)
plt.ylabel('accuracy', fontsize=16)
plt.plot(range(len(epoch_accuracy)),epoch_accuracy)
Model1.predict(inputs,labels,map_prediction=True)
plt.show()



acc=[]
pixel_count=63
for i in range(pixel_count):
    inputs1=Change_Pixel_Values(inputs,INPUT_LENGTH,i,data_type[input_type])
    print("After "+str(i)+" pixel has been changed")
    accuracy=Model1.predict(inputs1,labels,map_prediction=False)
    acc.append(accuracy)
plt.xlabel('Pixel Count', fontsize=18)
plt.ylabel('accuracy', fontsize=16)
plt.xlim(0, pixel_count), plt.ylim(0, 1.1)
plt.plot(range(pixel_count),acc)
plt.show()
