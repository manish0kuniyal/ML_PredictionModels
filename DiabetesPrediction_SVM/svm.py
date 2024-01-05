import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# female data
diabetes_dataset=pd.read_csv('diabetes.csv')

# print(diabetes_dataset.head())
# print(diabetes_dataset.shape)

# statistical data measure
# print(diabetes_dataset.describe())

# diabetic and non diabetic cases
print(diabetes_dataset['Outcome'].value_counts())
# print(diabetes_dataset['Age'].value_counts())

#mean for 0 and 1
print(diabetes_dataset.groupby('Outcome').mean())

#seperating data and labels
X=diabetes_dataset.drop(columns='Outcome',axis=1) #for column axis=0
Y=diabetes_dataset['Outcome']
# print(X)
# print(Y)

#Data Standardization - all values in the same range
scaler=StandardScaler()

scaler.fit(X)

standardized_data=scaler.transform(X)

# print(standardized_data)
X=standardized_data

# Splitting data into Train and Testing

X_train , X_test,Y_train ,Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# Training the model
classifier=svm.SVC(kernel='linear')

#training svm classifier
classifier.fit(X_train,Y_train)

#model evaluation
#accuracy on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('accuracy score -> ',training_data_accuracy)

#accuracy on the test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('accuracy for test -> ',test_data_accuracy)

#PREDICTION SYSTEM
input_data=(4,110,92,0,0,37,6,0,191,30)

#input to array
input_data_as_array=np.asarray(input_data)

#reshape the array
input_data_reshaped=input_data_as_array.reshape(1,-1)

# standardize the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)




