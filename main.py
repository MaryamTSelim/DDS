###############################################################
import pandas
dataset = pandas.read_csv('Iris.csv')
myInput = dataset.iloc[:,[1,2,3,4]].values
myOutput = dataset.iloc[:,[5]].values
print('Variables Imported')
##############################################################
from sklearn.preprocessing import LabelEncoder  , OneHotEncoder
lblEncoder = LabelEncoder()
myOutput[:,0] = lblEncoder.fit_transform(myOutput[:,0])
onehotEncoder = OneHotEncoder(categorical_features = [0])
myOutput = onehotEncoder.fit_transform(myOutput).toarray()
import numpy
myOutput = numpy.delete(myOutput,2,axis=1)
print('Executed LabelEncoder, OneHotEncoder and Numpy')
##############################################################
from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(myInput , myOutput , test_size = 0.2 , random_state = 0)
print('Variables Splited')
##############################################################
#Algorithms
##############################################################
#1- Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor().fit(xtrain,ytrain)
ypredictDT = regressor.predict(xtest)
###############################################################
#2- Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier().fit(xtrain, ytrain)
ypredictRF = forest.predict(xtest)
###############################################################
#2- K-Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier().fit(xtrain,ytrain) 
ypredictKNN = neigh.predict(xtest)
###############################################################
