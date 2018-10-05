import pandas as pd
from scipy.io import arff
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import neighbors
import math
import numpy as np

def preprocessCars(dataset):
    #Convert doors/persons coloumn to completely numeric
    dataset[[2]] = dataset[[2]].replace("5more",5)
    dataset[[3]] = dataset[[3]].replace("more",6)

    #save targets and convert to 1d numpy array 
    targets = dataset[[5]].values
    targets = targets[:,0]
    
    #remove unneeded data
    dataset = dataset.drop(columns=[5,6])
    
    #Use One Hot Encoding to separate catergorical data
    dataset = pd.get_dummies(dataset, columns=[0,1,4], prefix=["buying","maint","lug"])
     
    #Transform the data
    scaler = StandardScaler()
    newdataset = scaler.fit(dataset)
    normalDataset = newdataset.transform(dataset)
    
    return normalDataset,targets

def preprocessAutism(dataset):
    #Replace '?' with NaN
    i = 0
    while i < 20:  
        dataset[[i]] = dataset[[i]].replace('?', np.NaN)
        i = i + 1
    
    #Set up one hot encoding
    dataset = pd.get_dummies(dataset, columns=[11,12,13,14,15,16,18,19], prefix=["sex","race","jundice","autism","residence","app","age_des","relation"])
    dataset = dataset.dropna()
    
    #Save targets and convert to numpy array
    targets = dataset[[20]].values
    targets = targets[:,0]
    dataset = dataset.drop(columns=[20])
    
    #Transform the data
    scaler = StandardScaler()
    newdataset = scaler.fit(dataset)
    normalDataset = newdataset.transform(dataset)
    return normalDataset, targets

def preprocessMPG(dataset):
    #Remove unneeded data
    dataset = dataset.drop(columns=[8])
    
    #Replace '?' with NaN
    i = 0
    while i < 8:  
        dataset[[i]] = dataset[[i]].replace('?', np.NaN)
        i = i + 1
    
    #Separate discrete data with one hot encoding
    dataset = pd.get_dummies(dataset, columns=[1,6,7], prefix=["cyl","year","origin"])
    dataset = dataset.dropna()
    
    #Save targets 
    targets = dataset[[0]].values
    targets = targets[:,0]
    dataset = dataset.drop(columns=[0])

    #Transform the data
    scaler = StandardScaler()
    newdataset = scaler.fit(dataset)
    normaldataset = newdataset.transform(dataset)
    
    return normaldataset,targets
    
    
def main(argv):
    #load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    datasetCars = pd.read_csv(url,header=None)
    url = "C://Users/Carruth/Desktop/Autism-Adult-Data.csv"
    datasetAut = pd.read_csv(url, header=None, skipinitialspace=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    datasetMPG = pd.read_csv(url, header=None, delim_whitespace=True)


    #prepare the data for processing
    carTrain, carTarget = preprocessCars(datasetCars)
    autismTrain, autismTarget = preprocessAutism(datasetAut)
    mpgTrain, mpgTarget = preprocessMPG(datasetMPG)
    
    #Split data into separate sets
    carTrain, carTest, carTrainTarget, carTestTarget = train_test_split(carTrain, carTarget, test_size = 0.30)    
    autismTrain, autismTest, autismTrainTarget, autismTestTarget = train_test_split(autismTrain, autismTarget, test_size = 0.30)
    mpgTrain, mpgTest, mpgTrainTarget, mpgTestTarget = train_test_split(mpgTrain, mpgTarget, test_size = .30)
    
    #kNN classifier
    carClassifier = KNeighborsClassifier(n_neighbors = 50)
    carModel = carClassifier.fit(carTrain, carTrainTarget)
    carPrediction = carModel.predict(carTest)            

    autismClassifier = KNeighborsClassifier(n_neighbors = 50)
    autismModel = autismClassifier.fit(autismTrain, autismTrainTarget)
    autismPrediction = autismModel.predict(autismTest)

    mpgClassifier = neighbors.KNeighborsRegressor(30,weights="uniform")
    mpgModel = mpgClassifier.fit(mpgTrain, mpgTrainTarget)
    score = (mpgModel.score(mpgTest, mpgTestTarget))
                 
    #Display the results
    print ("Car Data Results")
    display(carPrediction, carTestTarget)
    print ("Autism Data Results")
    display(autismPrediction, autismTestTarget)
    print ("MPG Data Results")
    displayRegression (score)
    
    print ("\nCross Validation\n")
    mpgScores = cross_val_score(mpgModel,mpgTrain,mpgTrainTarget, cv=10)
    autismScores = cross_val_score(autismModel, autismTrain, autismTrainTarget, cv=10)
    carScores = cross_val_score(carModel, carTrain, carTrainTarget)
    print ("Cars: ", np.average(carScores))
    print ("Austism: ", np.average(autismScores))
    print ("MPG: ", np.average(mpgScores))
    
#Display Regression Data
def displayRegression(score):
    print (score * 100, " %")

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
        