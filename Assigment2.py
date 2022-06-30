#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data = pd.read_csv('assignment2_dataset_cars.csv')

x=data.iloc[:,0:3]
data.iloc[:,0:1]=le.fit_transform(data.iloc[:,0:1])

corr = data.corr()
#Top 50% Correlation training features with the Value
topfeature = corr.index[abs(corr['year'])>0.5]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[topfeature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

x = x[topfeature]

y=data.iloc[:,3:4]

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,shuffle=True,random_state=10)


poly_features = PolynomialFeatures(degree=1)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
predictionofY = poly_model.predict(poly_features.fit_transform(X_test))


print('MSE', metrics.mean_squared_error(y_test, predictionofY))


# In[ ]:





# In[ ]:




