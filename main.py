import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv("sonar.csv")

#sonar_data.head()


X= sonar_data.drop(columns =sonar_data.columns[-1], axis =1)
Y=sonar_data[sonar_data.columns[-1]]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y ,test_size =0.1 ,stratify =Y,random_state=1)

model=LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
training_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_accuracy)

X_test_prediction = model.predict(X_test)
test_accuracy=accuracy_score(X_test_prediction,Y_test)

print(test_accuracy)
