from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import math

def main(argv):
    iris = datasets.load_iris()

    #Split the data into training and test sets
    #Split the data into a training and testing set - keeping the targets aligned
    irisTrain, irisTest, targetTrain, targetTest = train_test_split(iris.data, iris.target,test_size = 0.30)
    
    #Transform data to get rid of curse of dimensionality
    scale = StandardScaler()
    scale2 = StandardScaler()
    
    trainingSet = scale.fit(irisTrain)
    testingSet = scale2.fit(irisTest)
    
    newIrisTrain = trainingSet.transform(irisTrain)
    newIrisTest = testingSet.transform(irisTest)
    
    classifier = kNN()
    
    #pass the training set to be part of the nearest neighbor object
    model = classifier.fit(newIrisTrain, targetTrain)
    #predict with test data and k set to 5
    prediction = model.predict(newIrisTest, k = 5)

    #Test against off the shelf algorithm
    newClassifier = KNeighborsClassifier(n_neighbors = 5)
    newModel = newClassifier.fit(irisTrain, targetTrain)
    newPrediction = newModel.predict(irisTest)
    
    print ("My algorithm")    
    display(prediction, targetTest)

    print ("Off the shelf algorithm")
    display(newPrediction, targetTest)
    
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


class kNN:
    def fit(self, data, targets):
        mod = nearest_neighbors(data, targets)
        return mod
        
class nearest_neighbors:
    def __init__(self, training = [], targets = []):
        self.irisT = training
        self.irisTar = targets
    def predict(self, data, k):
        #Prepare 3 arrays 
        results = []
        predictions = []
        k_array = []
        temp = 0
        #Loop through each row in the test set
        for rows in data:
            #Loop through all training data for each test row
            for row in self.irisT:
                #Calculate the distance between each of the 4 attributes
                #and add the result to the results array 
                dis1 = (rows[0] - row[0])**2
                dis2 = (rows[1] - row[1])**2
                dis3 = (rows[2] - row[2])**2
                dis4 = (rows[3] - row[3])**2
                #Array of the distance between 1 row of test data
                #and each row in the training set
                results.append(math.sqrt(dis1 + dis2 + dis3 + dis4))
            #This loop depends on the size of k
            while temp < k:
                #Find the index of the smallest distance in the results
                #array.
                ind = results.index(min(results))
                #Use the index to add the target value of the smallest
                #distance to the k_array
                k_array.append(self.irisTar[ind])
                #Change the minimum distance to the maximum to allow
                #the algorithm to find the next lowest distance if 
                #k allows for it.
                results[ind] = max(results)
                temp = temp + 1
            #Take the average of all the k_array elements to average
            #the results of the k nearest neighbors 
            avg = sum(k_array) / len(k_array)
            #Add the average to the predictions array as the guess for 
            #the classification of the data
            predictions.append(round(avg,0))
            k_array.clear()
            temp = 0
            results.clear()
        #Return the list of predictions
        return predictions

        
if __name__ == "__main__":
    main(0)
        