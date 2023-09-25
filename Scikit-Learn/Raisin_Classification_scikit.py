#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


# In[2]:


#Loading the data
print("Loading the data")
raisin_df = pd.read_excel('/Users/karan/Desktop/Karan/IITC/Academic/Spring 2022/Big Data/Project/BigDataProject/Raisin_Dataset/Raisin_Dataset.xlsx')


# In[3]:


#Defining the dataframe Schema, data types
print("Defining the dataframe Schema,data types")
print(raisin_df.info())


# In[4]:


#Shape of the dataframe
print("Dataframe shape: ", raisin_df.shape)


# In[5]:


#Top 5 rows of the dataframe
print("Top 5 rows of Dataframe")
print(raisin_df.head())


# In[6]:


#Summary statistics on every column
print("Summary Statistics on every column of dataframe:")
print(raisin_df.describe())


# In[7]:


#The number of null values present in each column
print("The number of null values present in each column:")
print(raisin_df.isnull().sum())


# In[8]:


#The pairwise plot of each column
sns.pairplot(raisin_df)
plt.show()


# In[9]:


print("The value counts of each target variable")
print(raisin_df['Class'].value_counts())


# In[10]:


#creating the barplot target value counts of each category:
print(raisin_df['Class'].value_counts().plot(kind = 'bar'))


# In[11]:


Y = raisin_df['Class']
X = raisin_df.drop('Class', axis = 1)


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


# In[13]:


corrMatrix = X_train.corr()
sns.set(rc = {'figure.figsize':(12,6)})
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[14]:


#Custom function that returns the columns which have high correlation values
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[15]:


corr_features = correlation(X_train, 0.85)
print("The below columns are dropped because they have a correlation value higher than the threshold of 0.85")
print(corr_features)


# In[16]:


X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)


# In[ ]:





# In[17]:


#Random Forest


# In[18]:


# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, Y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
from sklearn import metrics
print()

# using metrics module for accuracy calculation
print("Accuracy of the Random forest classifier model: ", metrics.accuracy_score(Y_test, y_pred))


# In[ ]:





# In[19]:


#Classifying the target class into numberical values
raisin_df.loc[raisin_df["Class"] == "Kecimen", "Class"] = 1
raisin_df.loc[raisin_df["Class"] == "Besni", "Class"] = -1


# In[20]:


Y = raisin_df['Class']
X = raisin_df.drop('Class', axis = 1)


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)


# In[22]:


corr_features = correlation(X_train, 0.85)


# In[23]:


X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)


# In[24]:


regressor= LinearRegression()
regressor.fit(X_train, Y_train)


# In[25]:


# Predicting y values 
y_pred = regressor.predict(X_test)


# In[26]:


y_pred_new = np.where(y_pred >= 0, 1, -1)


# In[27]:


errors = np.sum(y_pred_new != Y_test)


# In[28]:


print("The total number of test data:", (len(Y_test)))


# In[29]:


print("The number of wrongly classified values from test data:", errors)


# In[30]:


# Calculate and display accuracy
accuracy = 100 *((len(Y_test)-errors)/len(Y_test))
print('Accuracy of Linear Classifier Model:', round(accuracy, 2), '%.')


# In[ ]:





# In[ ]:




