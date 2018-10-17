import pandas as pd
from sklearn.model_selection import train_test_split
import math as m
import numpy as np
from sklearn import tree
from collections import Counter

def preprocess(dataset):
    #save targets and convert to 1d numpy array 
    targets = dataset[[5]].values
    targets = targets[:,0]

    #remove unneeded data
    dataset = dataset.drop(columns=[0,5])

    return dataset.values,targets

def main(argv):
    #load data
    #Trying to predict contacts
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data"
    dataset = pd.read_csv(url,header=None,delim_whitespace=True)

    #prepare the data for processing
    allData, allTargets = preprocess(dataset)
    
    #Split data into separate sets
    train, test, trainTarget, testTarget = train_test_split(allData, allTargets, test_size = 0.30)
    
    #ID3 classifier and make the tree
    classifier = ID3(train, trainTarget)
    model = classifier.makeTree(classifier.tree.root,classifier.data,classifier.classes,classifier.attributes,classifier.target)
    results = model.predict(test)
    
                     
    #Off the shelf version
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train,trainTarget)
    clf = clf.predict(test)
               
    #Display the results
    print ("My version from scratch: ")
    display(results, testTarget)
    print ("Sklearn version: ")
    display(clf, testTarget)
    
class ID3:
    def __init__(self, data = [], target = []):
        self.data = data
        self.target = target
        #Get unique elements from target array to see how many classes we have
        self.classes = self.unique(target)
        #Specific Attributes for lenses data
        self.attributes = ["Age","Perscription","Astigmatic","TearRate"]
        self.tree = Tree()
        
    def predict(self, test):
        #Initial variables
        predictions = []
               
        #Loop through each test case
        for item in test:
            tempI = item
            node = self.tree.root
            while (node.children is not None):
                node = self.traverse(node,tempI[int(node.name)])
                tempI = np.delete(tempI,int(node.parent.name))
            predictions.append(node.name)
        return predictions
    
    def traverse(self,node,branch):
        for child in node.children:    
            if (int(child.branchValue) == int(branch)):
                return child

    def makeTree(self, node, tempData, totalClasses, attributes, targetArray):
        # If there is only one label/target left, make a leaf
        if (len(self.unique(targetArray)) <= 1):
            if (len(self.unique(targetArray)) == 1):
                node.name = self.unique(targetArray)[0]
                node.children = None
            else:
                node.name = 2
                node.children = None
            return self
        
        # If there is no attribute to split on, make a leaf of most common target
        elif (len(attributes) == 0):
            cnt = Counter(targetArray)
            node.name = cnt.most_common(1)[0][0]
            node.children = None
            return self
            
        # Find a node to split on and recursively continue the process
        else:
            if (node is None):
                node = self.tree.root
            else:
                if (tempData is not None): 
                    tempData = np.delete(tempData,int(node.parent.name),1)
                        
            #Total entropy of the set
            entropyTotal = self.calcEntropy(None,self.data,self.target)      
            
            #Entropy if each column were the root respectively
            col = 0
            infoGain = []
            for i in tempData[0]:
                entropy = self.calcEntropy(col,self.data,self.target)
                col = col + 1
                infoGain.append(entropyTotal - entropy)
            #Which column will give the most info gain
            columnToSplitOn = infoGain.index(max(infoGain))
            
            #Set the node to the highest info gain
            if (node is None): 
                self.tree.root = Node(columnToSplitOn,None,self.attributes,None,[])
                node = self.tree.root
            else:
                node.name = columnToSplitOn
            
            #Create temp list of attributes to destroy slowly
            tempAtt = []
            for tem in node.availableAtt:
                tempAtt.append(tem)
            tempAtt.remove(node.availableAtt[int(node.name)])
            
            #Add child nodes 
            newNodeNames = self.unique(self.data[:,columnToSplitOn]) 
            for i in newNodeNames:
                newNode = Node("",str(i),tempAtt,node,[])
                node.children.append(newNode)

            print ("Node: ", node.availableAtt[int(node.name)], " ", node.name)
            for n in node.children:
                print ("Branch value from parent to child node: ", n.branchValue)
            
    
            #Repeat process for each child node to continue building
            #Trim the data and target array as the tree is growing
            newData = []
            newTarget = []
            targetCount = 0
            for n in node.children:
                for row in tempData:
                    if (int(row[int(node.name)]) == int(n.branchValue)):
                        newData.append(row)
                        newTarget.append(targetArray[targetCount])
                    targetCount = targetCount + 1
                #Recursive call to continue the tree. This will call all child nodes of each other node.
                self.makeTree(n,np.array(newData), self.classes, self.attributes, newTarget)
            
                #Reset temp arrays and counts for next recursive iteration
                newData.clear()
                newTarget.clear()
                targetCount = 0
            return self
            
    
    #Returns an array of the unique elements of the array parameter
    def unique(self, array):
        temp = []
        for i in array:
            if i not in temp:
                temp.append(i)
        return temp        
        
    #Calculates entropy for entire set or for each attribute
    def calcEntropy(self,column,data,targets):
        #Calculate entropy for entire dataset
        if (column is None):
            #Get counts of all the elements
            countArray = Counter(self.target)
            
            #Calculate entropy
            entropyCalc = None
            length = len(self.target)
            for i in self.classes:
                # ratio is p in plog2(p)
                ratio = countArray[i] / length
            
                #This if/else statement assures the correct summing of the entropy
                if (entropyCalc is None):
                    entropyCalc = ratio * m.log2(ratio)
                else:
                    entropyCalc = entropyCalc - ratio * m.log2(ratio)
            return entropyCalc 
        
        #Calculate entropy based on choosing a certain attribute
        else:
            #How many options are there in the column
            options = self.unique(data[:,column])
            #Calculate what classes are present for given attribute
            index = 0
            count = 0
            counts = []
            entropyForEachAttribute = []
            classes = []
            #Loop through each possible attribute in options
            for att in options:
                #Loop through each row looking for the correct attribute
                for row in data:
                    #If attributes match, add to class array
                    if (att == row[column]):
                        classes.append(self.target[index])
                        count = count + 1
                    index = index + 1
                #Keep track of how many elements go to each node
                counts.append(count)
                #count how many of each class would be in this attribute's node
                sums = Counter(classes)
                #Calculate entropy 
                entr = None
                for i in self.classes:
                    # ratio is p in plog2(p)
                    ratio = sums[i] / count
                    #This if/else statement assures the correct summing of the entropy
                    if (ratio == 0.0):
                        entr = entr
                    elif (entr is None):
                        entr = ratio * m.log2(ratio)
                    else:
                        entr = entr - ratio * m.log2(ratio)
                    
                #Keep track of the entropies for each node
                entropyForEachAttribute.append(entr)
                classes.clear()
                index = 0
                count = 0
            
            #Weighted entropy calculation  
            ind = 0
            finEntropy = 0
            for ent in entropyForEachAttribute:
                finEntropy = finEntropy + (ent * counts[ind] / len(self.target))                                                    
        
            return finEntropy
        
class Tree:
    def __init__(self, root = None):
        self.root = root
        
    def display(self):
        return self

class Node:
    def __init__(self, name = "", branchValue = "", availableAtt = [], parent = None, children = []):
        self.name = name
        self.parent = parent
        self.availableAtt = availableAtt
        self.children = children
        self.branchValue = branchValue

class Leaf:
    def __init__(self, name=-1):
        self.name = name
        
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
        