import numpy as np

class layer():
    def __init__(self,name,type,nodes_number):
        self.name=name
        self.type=type
        self.nodes_number=nodes_number
        self.input_values=np.zeros(shape=(nodes_number,1),dtype=float)##input values of nodes
        self.sum_values=np.zeros(shape=(nodes_number,1),dtype=float)##sum values of nodes
        self.output_values=np.zeros(shape=(nodes_number,1),dtype=float)##output values of nodes


    def set_input_values(self,input):
        self.input_values=input
        if (self.type=="input"):
            self.set_output_values(input)

    def set_output_values(self,output):
        self.output_values=output




class Model():
    def __init__(self,method,input_type,perceptron_rule):
        self.method=method##method
        self.perceptron_rule=perceptron_rule
        self.layers=[]##layers of Model
        self.input_type=input_type
        """For Training  """
        self.Connections_Weight=[]## weight of Connections are stored
        self.Connections_Bias=[]##Bias of Connections are stored
        self.input_number=0##total input number for training model, using for iteration during epoch state
        self.input_length=0##each input's length also output array length
        self.input_arr=0##input array
        self.output_arr=0##output array
        self.output_length=0##output length

    def add_layer(self,layer):
        self.layers.append(layer)


    def create_weight_and_bias_array(self,layer1,layer2,bias):
        ##create arrays as correspond to connections with layers nodes number
        w_array=np.zeros(shape=(layer1.nodes_number,layer2.nodes_number),dtype=float)
        self.Connections_Weight.append(w_array)##append to model weight list
        b_array=np.full(shape=(layer2.nodes_number),fill_value=float(bias))
        self.Connections_Bias.append(b_array)


    def set_input_values(self,input_arr,input_number,input_length):
        if(type(input_arr)!=np.ndarray):
            raise Exception("Type Error: given input aren't ndarray")
        input_layer=self.layers[0]
        if not(input_length==input_layer.input_values.shape[0]):
            raise Exception("input's length and nodes number of input layer aren't matched")

        self.input_number=input_number
        self.input_length=input_length
        self.input_arr=input_arr

    def set_output_values(self,output_arr,output_length):
        if(type(output_arr)!=np.ndarray):
            raise Exception("Type Error: given output aren't ndarray")

        output_layer=self.layers[-1]
        if not(output_length==output_layer.output_values.shape[0]):
            raise Exception("output's length and nodes number of output layer aren't matched")

        self.output_length=output_length
        self.output_arr=output_arr

    def activation_func(self,y_in,th):
        y=1.0
        if (-th < y_in < th):
            y=0
        elif (y_in<-th):
            y=-1.0
        return y

    def activation_func_bin(self,y_in,th):
        y=1.0
        if (y_in < th):
            y=0
        return y

    def default_rule(self,input_arr,out,w_array,b_array,n,j):
        for k,inp in enumerate(input_arr):##Update weights
            w_array[k][j]=w_array[k][j]+n*out*inp
        b_array[j]=b_array[j]+n*out##Update bias value

    def delta_rule(self,input_arr,out,w_array,b_array,n,j,y):
        for k,inp in enumerate(input_arr):##Update weights
            w_array[k][j]=w_array[k][j]+n*(out-y)*inp
        b_array[j]=b_array[j]+n*(out-y)##Update bias value


    def Feed_Forward_Perceptron(self,input_arr,output_arr,n,th):
        #bool=np.full((input_layer.nodes_number,output_layer.nodes_number),False)##boolean matrix for weight values
        #while bool.all()!=True:##Until weights for each connections maintaing equation
        w_array=self.Connections_Weight[0]
        b_array=self.Connections_Bias[0]
        y=0
        for j,out in enumerate(output_arr):

            y_in=0## sum
            for i,inp in enumerate(input_arr):
                y_in+=inp*w_array[i][j]
            y_in+=b_array[j]##bias

            if(self.input_type=="binary"):##activation
                y=self.activation_func_bin(y_in,th)
            elif(self.input_type=="bipolar"):
                y=self.activation_func(y_in,th)

            if(y!=out):
                if self.perceptron_rule == "default":
                    self.default_rule(input_arr,out,w_array,b_array,n,j)
                if self.perceptron_rule == "delta":
                    self.delta_rule(input_arr,out,w_array,b_array,n,j,y)

    def Perceptron(self,learning_rate,epoch,threshold,bias):
        iter=0
        self.create_weight_and_bias_array(self.layers[0],self.layers[1],bias)#give input and output layer as arguments
        acc=[]
        while iter!=epoch:
            for i in range(self.input_number):
                self.Feed_Forward_Perceptron(self.input_arr[i],self.output_arr[i],learning_rate,threshold)
            iter+=1
            if(iter%1==0):
                print("epoch="+str(iter))
                accuracy=self.predict(self.input_arr,self.output_arr,map_prediction=False)
                acc.append(accuracy)
        return acc
        #print("!!!Weights Matrix After Training!!!"+str(self.input_length)+"X"+str(self.output_length))
        #print(self.Connections_Weight[0])

    def train(self,learning_rate,epoch,bias,threshold):#return accuracy value of each epoch
        if self.method=="perceptron":
            acc=self.Perceptron(learning_rate,epoch,threshold,bias)
        return acc

    def predict_per_once(self,input,output):##predict a input
        w_array=self.Connections_Weight[0]
        b_array=self.Connections_Bias[0]
        pred_result=np.zeros(shape=(self.output_length),dtype=np.float64)
        for j,out in enumerate(output):
            y_in=0.0
            for i,inp in enumerate(input):
                w=w_array[i][j]
                y_in+=inp*w_array[i][j]
            y_in+=b_array[j]
            pred_result[j]=int(y_in)

        return pred_result

    def Map_Pred_Matrix(self,results):##listing predictions on matrix with pred value as x, real value as y
        print("""!!!!!!!!Results Of Prediction Of Given Inputs!!!!!!!!""")
        sep=" | "
        Letters=["L","A","B","C","D","E","J","K"]
        l=sep.join(map(str,Letters))
        print("\t"+l)
        for i,row in enumerate(results):
            print("\t-----------------------------")
            x=sep.join(map(str,row))
            print("\t"+Letters[i+1]+" | "+x)

    def predict(self,inputs,labels,map_prediction):##array that have more than one input as argument
        true_result=0
        false_result=0
        results=[[0 for x in range(self.output_length)] for x in range(self.output_length)]
        for i,input in enumerate(inputs):
            pred_result=self.predict_per_once(input,labels[i])
            pred_class=np.argmax(pred_result)##return index of max value as predicted class
            real_class=np.where(labels[i]==1)[0][0]
            results[pred_class][real_class]+=1
            if pred_class==real_class:
                true_result+=1
            else:
                false_result+=1
        if(map_prediction==True):
            self.Map_Pred_Matrix(results)
        accuracy=float(true_result) / float(true_result+false_result)
        print("accuracy=>"+str(accuracy))
        return accuracy
