# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required Python libraries and create the datasets with study hours and marks.
2.Divide the datasets into training and testing sets.
3.Create a simple Linear Regression model and train it using the training data.
4.Use the trained model to predict marks on the testing data and display the predicted output.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Bristo AK
RegisterNumber:212225230037
*/
```
```
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')
model.fit(X, y)
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.show()
 
```

## Output:
![alt text](<Screenshot 2026-01-30 141221.png>)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
