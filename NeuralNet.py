import numpy as np
import math as m
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv

def main(arg):
    
    #Load dataset
    data = datasets.load_iris()
    neuronTarget = np.unique(data.target)
    targets = data.target
    data = np.array(data.data)
    
    #Transform data to get rid of curse of dimensionality
    scale = StandardScaler()
    newData = scale.fit(data)
    data = newData.transform(data)
    
    
    #Separate the data into training and testing
    train, test, targetTrain, targetTest = train_test_split(data, targets,test_size = 0.30)
    
    #Set up layer variables for network
    numLayers = 0
    numNodesInHiddenLayers = [3]
    numOutputNodes = len(neuronTarget)
    biasArray = -(np.ones(shape=(len(train),1)))

    #Create a Neural Network by looping through each layer and 
    #create random weights for each layer
    nNet = NeuralNet()
    #Initial weights from inputs to first layer
    if (numLayers == 0):
        weights = makeLayer(inputs = train.shape[1], nodes = numOutputNodes)
    else:
        weights = makeLayer(inputs = train.shape[1], nodes = numNodesInHiddenLayers[0])

    nNet.weights.append(weights)

    #Index for numNodes array to acces the n + 1 element 
    ind = 0
    if (numLayers > 0):  
        for n in numNodesInHiddenLayers:        
            #If we are not at the last layer, continue making matricies
            if (numLayers > 1):
                tempWeights = makeLayer(inputs = n, nodes = numNodesInHiddenLayers[ind + 1])     
            #Last layer needs to have proper amount of output nodes
            else:
                tempWeights = makeLayer(inputs = n, nodes = numOutputNodes)
            nNet.weights.append(tempWeights)
            ind += 1
            numLayers -= 1
            
    #Train the network
    # I is the number of training iterations
    newTrain = np.column_stack((train, biasArray))
    i = 0
    while (i < 3000):
        tempTargets = nNet.train(newTrain,targetTrain)
        i += 1
        

    print ("Number of trials: ", i)
    #display the results
    display(tempTargets, targetTrain) 

    

#MakeLayer returns a matrix of randomly generated weights based on the 
#given input number and node number.
def makeLayer(inputs, nodes):
    #Account for a bais nodes with individual weights
    bias = inputs + 1
    
    #Initialize random weights for each input to each node
    #Random # between -0.5 and .5
    weights = np.random.uniform(low=-.5,high=.5,size=(bias,nodes))
    
    return weights
    
class NeuralNet:
    def __init__(self, weights = [], learningRate = .2):
        #Array of multi dimensional arrays for weights at each layer
        self.weights = weights
        self.learningRate = learningRate

    #This function calculate activations for a multi dimension array
    def activationFunction(self, value):
        #Return array of all activations
        arrayOfTotals = []
        for v in value:
            arrayOfActivations = []
            for i in v:    
                #Sigmoid function
                temp = 1 / (1 + (m.e ** -(i)))
                arrayOfActivations.append(temp)
            arrayOfTotals.append(arrayOfActivations)
        return arrayOfTotals
    
    #Round the results of the output nodes
    def roundData(self, data):
        totals = []
        for d in data:
            temps = []
            for i in d:
                temps.append(round(i))
            totals.append(temps)
        return totals

    def train(self, data, targets): 
        #loop through the data doing some matrix math
        #prediction container
        predictions = []
        allActivations = []
        allActivations.append(data)
        #Dot product of the entire data set against the first layer
        dot = data.dot(self.weights[0])
        neuronOutput = self.activationFunction(dot)
        networkOutput = neuronOutput
        #loop through all of the layers of the network doing dot products
        index = 1
        while (index < len(self.weights)):      
            biasArray = -(np.ones(shape=(len(neuronOutput),1)))
            neuronOutput = np.column_stack((neuronOutput, biasArray))
            dot = neuronOutput.dot(self.weights[index])
            neuronOutput = self.activationFunction(dot)
            allActivations.append(np.column_stack((neuronOutput,biasArray)))
            index += 1
        
        allActivations.append(networkOutput)
        #Calculate the error at each layer starting with ouptut layer     
        errors = []
        length = len(allActivations) - 1
        error, sumOfErrors = self.calculateErrorOutput(allActivations[length - 1], allActivations[length], targets)
        #temp, temp2 = self.calError(allActivations[length - 2], allActivations[length - 1], sumOfErrors)
        errors.append(error)
        #errors.append(error)
       
# You successfully propogate one layer. Now you need to loop through each
# layer backwards. Loop through each and call the propagate
# function from here looping through the weights. Pass it the weights and error.
# After that, write your predict function
        
        #Back propagation through the network
        self.propagate(errors)
          
        #Round the results to get 0 or 1
        allPredictions = self.roundData(neuronOutput)
    
        return allPredictions

    #Calculate error for hidden layers    
    def calError(self, prev, activations, errorFromLayer):
        sumOfError = 0
        index = 0
        index2 = 0
        activations = np.delete(activations,len(activations[0])-1,1)
        total = np.zeros(shape=(len(activations[0]),len(prev[0])))
        for n in activations:
            ind = 0
            for i in n:
                error = []
                for hope in prev[index2]:
                    error.append(errorFromLayer * i * (1 - i))
                    l = len(self.weights) - 1 
                for x in np.transpose(self.weights[0])[ind]:
                    sumOfError += errorFromLayer * i * (1 - i) * x
                ind += 1
            total[index] += np.array(error)             
            index += 1
            index %= len(activations[0]) - 1
            index2 += 1

        total /= len(activations)
        sumOfError /= len(activations)

        return total, sumOfError
    
        #We have a sum of all errors, now we take the average over the whole set
        total /= len(activations)
        sumOfError /= len(activations)
        return total, sumOfError

    #Back propagation 
    def propagate(self, err):
        #Do it for each weight
        count = 0
        for w in self.weights:
            tempWeight = self.weights[count] - np.transpose(err[count])
            self.weights[count] = np.array(tempWeight)
            count += 1

    #Calculate the error at the output nodes
    def calculateErrorOutput(self, data, predict, targ):
        #Loop through all predctions and compare targets
        index = 0
        sumOfError = 0
        total = np.zeros(shape=(len(predict[0]),len(data[0])))
        for p in predict:
            if (targ[index] == 0):
                correctOutput = [0,1,0]
            elif (targ[index] == 1):
                correctOutput = [0,1,0]
            else:
                correctOutput = [0,0,1]
            ind = 0
            for i in p:
                error = []
                for d in data[index]:
                    err = i * (1 - i) * (i - correctOutput[ind])
                    error.append(err * self.learningRate * d)
                l = len(self.weights) - 1 
                for x in np.transpose(self.weights[l])[ind]:
                    sumOfError += err * x
                total[ind] += np.array(error)             
                ind += 1
            index += 1
        #We have a sum of all errors, now we take the average over the whole set
        total /= len(predict)
        sumOfError /= len(predict)
        return total, sumOfError

    def predict(self, data):
        return 0



#Display the accuracy of the kNN algorithm
def display(prediction, targetTest):
    count = 0
    wrong = 0

    while count < len(targetTest):
        if (targetTest[count] == 0):
            correctOutput = [0,1,0]
        elif (targetTest[count] == 1):
            correctOutput = [0,1,0]
        else:
            correctOutput = [0,0,1]

        #compare the prediction array with the target array
        if prediction[count] != correctOutput:
            wrong = wrong + 1
        count = count + 1
    print (count - wrong, "/", count, " ", "{:.2f}".format(((count - wrong)/count) * 100), "%")







if __name__ == "__main__":
    main(0)