import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    
    #Make the network with initial random weights
    weights, neurons = makeNetwork(inputs = train.shape[1], nodes = len(neuronTarget))
    
    #Create the classifier passing the weights matrix and the neuron thresholds
    classifier = NeuralNet(weights, neurons)
    
    #Train the network
    #This temoraraly predicts for week 1
    tempTargets = classifier.train(train,neuronTarget)
    
    #display the results
    display(tempTargets, targetTrain)

def makeNetwork(inputs, nodes):
    #Account for a bias node for each node in layer 1
    #This will add to the number of inputs so that each
    #bias node has an individual weight
    bias = inputs + 1
    
    #Initialize random weights for each input to each node
    #Random # between -0.5 and .5
    weights = np.random.uniform(low=-.5,high=.5,size=(bias,nodes))
    
    #Initialize nodes with threshold = 0
    neurons = np.zeros(nodes)
    
    return weights, neurons
    
class NeuralNet:
    def __init__(self, weights = None, neurons = None):
        self.weights = weights
        self.neurons = neurons

    def train(self, data, targets): 
        #loop through the data doing some matrix math
        #prediction container
        predictions = []
        for row in data:
            #Add a bias input
            row = np.append(row,[-1])
            
            #Dot product of the transpose of the weights matrix and the inputs
            #This is the same as multiplying all the weights and inputs for 
            #Each neuron and adding them all together in one line of code
            dot = np.transpose(self.weights).dot(row)
            
            #loop through each neuron value comparing thresholds
            index = 0
            count = 0
            for n in self.neurons:
                if (dot[index] > n):
                    count += 1
                index += 1
            #Add to the predictions
            predictions.append(count % len(self.neurons))
        return predictions
    
    def predict(self, data):
        return 0



#Display the accuracy of the kNN algorithm
def display(prediction, targetTest):
    count = 0
    wrong = 0

    while count < len(targetTest):
        #compare the prediction array with the target array
        if prediction[count] != targetTest[count]:
            wrong = wrong + 1
        count = count + 1
    print (count - wrong, "/", count, " ", "{:.2f}".format(((count - wrong)/count) * 100), "%")








if __name__ == "__main__":
    main(0)