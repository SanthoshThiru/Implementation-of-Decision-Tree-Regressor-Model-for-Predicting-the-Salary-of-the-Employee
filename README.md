# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step-1:
Import the required libraries .
### Step-2:
Read the data frame using pandas.
### Step-3:
Get the information regarding the null values present in the dataframe.
### Step-4:
Apply label encoder to the non-numerical column inoreder to convert into numerical values.
### Step-5:
Determine training and test data set.
### Step-6:
Apply decision tree regression on to the dataframe.
### Step-7:
Get the values of Mean square error, r2 and data prediction.

## Program:
```/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Santhosh T
RegisterNumber: 212223220100
*/


import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

### data.head()
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/fe2c1be3-b3ab-4739-acef-2b491a32342f)

### data.info()
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/dd863ff5-7de2-4492-96e4-4ec89d70dd57)


### isnull() & sum() function
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/a8b4cad7-235e-45b8-b65d-6225a7479851)


### data.head() for position
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/2bd6d3c9-cfc2-4691-af28-b5990cb0f800)


### MSE value
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/9dc4af78-4795-4e1c-8de8-02cfc6ff6255)


### R2 value
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/295605dd-34ab-4ec8-92e7-cc8acbe3fdc5)


### Prediction value
![image](https://github.com/SanthoshThiru/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/148958618/4aca6625-e3ff-4228-8f0d-8f6f253ff712)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
