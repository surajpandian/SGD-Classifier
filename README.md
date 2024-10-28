# SGD-Classifier
### Date:
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the Program.

Step 2: Import Necessary Libraries and Load Data

Step 3: Split Dataset into Training and Testing Sets

Step 4: Train the Model Using Stochastic Gradient Descent (SGD)

Step 5: Make Predictions and Evaluate Accuracy

Step 6: Generate Confusion Matrix

Step 7: Stop the program.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: R suraj pandian
RegisterNumber:  212223080040

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
iris = load_iris()

#create pandas dataframe
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

print(df.head())

#split the data into features x and target y
x=df.drop('target',axis=1)
y= df['target']

#split the data into training and testing set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

sgd_clf = SGDClassifier(max_iter = 1000, tol= 1e-3)#create sgd classifier with default parameter
sgd_clf.fit(x_train, y_train)#train the classifier on the training data

y_pred = sgd_clf.predict(x_test)#make predictions on the testing data

accuracy = accuracy_score(y_test, y_pred)# evaluate the classifier's accuracy
print(f"Accuracy:{accuracy:.3f}")

#calculate the confusion matrix
cf=confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cf)
*/
```

## Output:
### Data:
![71](https://github.com/user-attachments/assets/eb4171f5-9f03-48a7-b788-b5454cb1c0ac)

### Accuracy:
![72](https://github.com/user-attachments/assets/2239e11c-61e7-48b8-b786-8b02a95ac1d9)

### Matrix:
![73](https://github.com/user-attachments/assets/ac5fc365-639d-4d1b-9f16-dd2fd31f4f83)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
