import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_mail_data=pd.read_csv("mail_data.csv")
# print(raw_mail_data)

#replacing null values with null strings
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# print(mail_data.head())

# print(mail_data.shape)

mail_data.loc[mail_data["Category"]=="spam",'Category']=0
mail_data.loc[mail_data["Category"]=="ham",'Category']=1


x=mail_data['Message']
y=mail_data['Category']
# print(x)
# print(y)

x_trian,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=3)

#feature extraction
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
x_train_features=feature_extraction.fit_transform(x_trian)
x_test_features=feature_extraction.transform(x_test)


y_train=y_train.astype('int')
y_test=y_test.astype('int')

#Logistic Regression
model = LogisticRegression()
model.fit(x_train_features,y_train)

#Prediction on training data
prediction_on_training=model.predict(x_train_features)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training)

print("ACCURACY - " ,accuracy_on_training_data)