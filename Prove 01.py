from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Requirement 1 - Load the data
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

#Requirement 2 - Split the data into training and test sets
#Split the data into a training and testing set - keeping the targets aligned
irisTrain, irisTest, targetTrain, targetTest = train_test_split(iris.data, iris.target,test_size = 0.30)

#Requirement 3
classifier = GaussianNB()
model = classifier.fit(irisTrain, targetTrain)

#Requirement 4
predicted = model.predict(irisTest)

count = 0
wrong = 0
while count < len(targetTest):
    if predicted[count] != targetTest[count]:
        wrong = wrong + 1
    count = count + 1
print (count - wrong, "/", count, " ", "{:.2f}".format(((count - wrong)/count) * 100), "%")

#Requirement 5
class HardCodedClassifier:
    def fit(self, data, targets):
        mod = HardCodedModel()
        return mod
        
class HardCodedModel:
    def predict(self, data):
        i = 0
        while i < len(data):
            data[i] = i % 2
            i = i + 1
        return data

def main(argv):
    classifier = HardCodedClassifier()
    model2 = classifier.fit(irisTrain, targetTrain)
    pred = model2.predict(irisTest)
    count = 0
    wrong = 0
    while count < len(targetTest):
        if pred[count,0] != targetTest[count]:
            wrong = wrong + 1
        count = count + 1
    print (count - wrong, "/", count, " ", "{:.2f}".format(((count - wrong)/count) * 100), "%")

        
main(0)
        
        