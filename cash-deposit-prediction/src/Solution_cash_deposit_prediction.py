# -*- coding: utf-8 -*-
"""
Created on Mon Mar 01 09:00:30 2021

@author: Nikita Porwal
#Email - porwal.nikit20@gmail.com
#Contact - +91 9424847351

Project Title: Cash Deposit Prediction

Steps followed for prediction:

1. Importing Library and dataset
2. Data Sanity Check and Data Cleaning
3. Class/Method is created for repeated functions
4. Imputing Missing Values
5. Adding New Features here it is Establishment_Year
6. Data Label Encoding
7. Outliers Detection using Boxplot and managing outliers
8. Checking Linearity in data
9. Creating histrogram to check distribution of data
10. Model selection:
    As we can see the dataset is of continous numeric type and Supervised ML data
    where we will use Regression Algorithms,
    Tried various models to check the best fit for data prediction:
    i. Used Decision Tree Regression Model for prediction
    ii. Used Linear Regression Model for prediction
    iii. Used Random Forest Regression Model for prediction
    iv. Used KNeighbor Regression Model for prediction
    v. Gride Search CV using Ridge Model for prediction
 11. Comparing Accuracy and RMSE values to check for best fit model    
"""

#####CASH DESPOSIT PREDICTION####

#Import Required Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Code starts here
cash_train = pd.read_csv(r"C:\Users\Rohit\Imarticus\Python\Projects\Capstone_Project\Dataset\train.csv",header=0)
cash_test = pd.read_csv(r"C:\Users\Rohit\Imarticus\Python\Projects\Capstone_Project\Dataset\test.csv",header=0)
#Basic Data Sanity Check
print(cash_train.shape)
print(cash_test.shape)

print(cash_train.columns)
print(cash_test.columns)

#Making Columns Similar to clean both Test and Train data in parallel
print(cash_train.columns.difference(cash_test.columns))
#deposit_amount_2017 is the extra column and is needs to be predicted
# Code ends here

# Code starts here
cash_train.drop(['deposit_amount_2017'],axis = 1,inplace = True)


#Creating a column to differenciate between Train and Test
cash_train['Sample'] = 'train'
cash_test['Sample'] = 'test'

data = pd.concat([cash_train,cash_test],axis=0,sort=False) #Combine the data

#Check the Data
data.head()
data.info()

data.replace('NaN', np.nan)
#check the data type of each variable in dataframe
data.dtypes
#here we can see Date of estabishment has different data type so convert it to datetime 

data['date_of_establishment'] = pd.to_datetime(data['date_of_establishment'])
data.dtypes #seems perfect now
# Code ends here

# Code starts here
#--------------------------------------------------------------------------
#Created a class for most repaeated required tasks
#check missing values in data and columns along with their percentage
#remove columns from dataframe
#check count of NA for each column

class Data_Auditor:
    def NA_in_Data(data_frame):
        result = (data_frame.isnull().sum().sum()) / (data_frame.shape[0] * data_frame.shape[1]) * 100
        return(print("Data has",round(result,2),"% NA's"))
    
    def Remove_Columns(data_frame,*args):
        list_of_cols = list(args)
        data_frame.drop(list_of_cols,axis = 1,inplace = True)
        
    def NA_in_Columns(data_frame):
        total_missing = data_frame.isnull().sum().sort_values(ascending=False)
        percent_missing = round(((data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)*100),1)
        missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Missing_Obs', 'Percent_of_NA'])
        return(missing_data.head(10))

Data_Auditor.NA_in_Data(data) #There are 6.75% of NA's we need to investigate further
Data_Auditor.NA_in_Columns(data) #Lets look at it columns wise
# Code ends here

# Code starts here
##IMPUTING OUT THE MISSING DATA##
#For the deposit amount in columns where we have missing values lets replace it with mean
for column in ['deposit_amount_2011','deposit_amount_2012','deposit_amount_2013','deposit_amount_2014','deposit_amount_2015','deposit_amount_2016']:
    data[column].fillna(data[column].mean(), inplace=True)

Data_Auditor.NA_in_Data(data) #Now we can see only 3.77% missing data is there
Data_Auditor.NA_in_Columns(data) #only Date column contains missing values

data['date_of_establishment'] = data['date_of_establishment'].fillna("")

Data_Auditor.NA_in_Data(data) #There is 0% of NA's now
Data_Auditor.NA_in_Columns(data) #Lets look at it columns wise
#No missing value
# Code ends here

# Code starts here
#Feature Engineering
#Remove useless column Id
Data_Auditor.Remove_Columns(data,'id')
data.reset_index(drop=True,inplace=True)

#Adding new feature as Year of Establishment
data['Year_Established'] = pd.DatetimeIndex(data['date_of_establishment']).year
data['Year_Established'] = data['Year_Established'].fillna("")

#Removing the Date column as we don't need it now
Data_Auditor.Remove_Columns(data,'date_of_establishment')

data.head(10)
# Code ends here

# Code starts here
#DATA CONVERSION
#Encoding some categorical variables into Ordered Numeric Variable as they are ranked
#variables
data.columns

from sklearn.preprocessing import LabelEncoder
columns =('location.Code', 'location','loc.details','state','Sample')

# process columns, apply LabelEncoder to categorical features

for cols in columns:
    lbl = LabelEncoder() 
    lbl.fit(list(data[cols].values)) 
    data[cols] = lbl.transform(list(data[cols].values))

#Dummy Value Encoding
data = pd.get_dummies(data)
print(data.shape) #data output (5413 rows, 168 columns) from (5413 rows, 14 columns)

#Split Test and Train Data
#Train
Train_data = data[data['Sample'] == 1]
del Train_data['Sample']
df = pd.read_csv(r"C:\Users\Rohit\Imarticus\Python\Projects\Capstone_Project\Dataset\train.csv")
Train_data = pd.concat([Train_data, df['deposit_amount_2017']], axis=1).reindex(Train_data.index)
Train_data['deposit_amount_2017'] = Train_data['deposit_amount_2017'].fillna(0)

#Assumption 1:There should be no outliers in the data
#Lets check for outliers
#boxplot
# Distribution of univariate data over the median also helps to detect the outliers
plt.figure(figsize=(10,10))
outlier_check = ['deposit_amount_2011','deposit_amount_2012','deposit_amount_2013','deposit_amount_2014','deposit_amount_2015','deposit_amount_2016']
sns.boxplot(data=Train_data[outlier_check])

#Removing Outliers
Train_data = Train_data.drop(Train_data[(Train_data['deposit_amount_2011']>200000) | (Train_data['deposit_amount_2012']>200000) | (Train_data['deposit_amount_2013']>200000)
| (Train_data['deposit_amount_2014']>200000) | (Train_data['deposit_amount_2015']>200000) | (Train_data['deposit_amount_2016']>200000)].index)

#Lets check the boxplot again
plt.figure(figsize=(10,10))
outlier_check = ['deposit_amount_2011','deposit_amount_2012','deposit_amount_2013','deposit_amount_2014','deposit_amount_2015','deposit_amount_2016']
sns.boxplot(data=Train_data[outlier_check])
#seems quiet well

#Test
Test_data = data[data['Sample'] == 0]
del Test_data['Sample']


#Creating Train Test Data Split for Model
X = Train_data[Train_data.columns[Train_data.columns != 'deposit_amount_2017']]
y = Train_data[Train_data.columns[Train_data.columns == 'deposit_amount_2017']]

X_test_original = Test_data[Test_data.columns[Test_data.columns != 'deposit_amount_2017']]

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# split into 70:30 ration 
X_train_original, X_val_original, y_train_original, y_val_original = train_test_split(StandardScaler().fit_transform(X), y, test_size = 0.1, random_state = 0) 
#y_val_original.reset_index(drop=True,inplace=True)

X_train = X_train_original
y_train = y_train_original
X_val = X_val_original
y_val = y_val_original

X_test = X_test_original
# Code ends here

# Code starts here
#Assumption 2: Assumption of Linearity:Every independent variable should have a linear relationship with the dep var
sns.pairplot(Train_data,x_vars=["deposit_amount_2011","deposit_amount_2012","deposit_amount_2013",
                               "deposit_amount_2014","deposit_amount_2015","deposit_amount_2016"],y_vars="deposit_amount_2017",kind='reg')

# Code ends here

# Code starts here
#Assumption 3: Dependant variable should follow an approx. normal distribution
sns.distplot(y,color='b',bins=10,hist = True)
# Code ends here
    
# Code starts here
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X_train = X_train_original
y_train = y_train_original
X_val = X_val_original
y_val = y_val_original

X_test = X_test_original

dt = DecisionTreeRegressor(random_state=5)

dt.fit(X_train,y_train)

accuracy_dt = dt.score(X_val,y_val)
print("\nAccuracy of Decision Tree",accuracy_dt)
y_pred_dt1 = dt.predict(X_val)
rmse_dt = np.sqrt(mean_squared_error(y_val,y_pred_dt1))
#print("y_pred:",y_pred_dt1)
#print("y_train",y_train)
print("\nDecision Tree Root Mean Square Error:",rmse_dt)

y_pred_dt = dt.predict(X_test)
cash_test['deposit_amount_2017'] = y_pred_dt

#Exporting required data into csv
#req_col = [0,14]
#cash_test.iloc[:,req_col].to_csv('cash_deposit_2017.csv',index=False)

#Visualising Decision Tree results
plt.plot(y_train,dt.predict(X_train),color='blue')
plt.title('Regression Tree Model')
plt.xlabel('Predicted Deposit Amount 2017')
plt.ylabel('Deposit Amount 2017')
plt.show()
# Code ends here

# Code starts here
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train = X_train_original
y_train = y_train_original
X_val = X_val_original
y_val = y_val_original

X_test = X_test_original

lr = LinearRegression()

lr.fit(X_train,y_train)
r_sq = lr.score(X_train,y_train) #R-Square

y_pred_lr = lr.predict(X_val)
accuracy_lr = lr.score(X_val,y_val)
print("\nAccuracy Linear regression:", accuracy_lr)
rmse_lr = np.sqrt(mean_squared_error(y_val,y_pred_lr))
print("\nLinear regression : Root Mean Square Error:",rmse_lr)

#Actual predictions
y_pred_lr = lr.predict(X_test)
#y_pred_lr = y_pred_lr[:,-1:]
cash_test['deposit_amount_2017'] = y_pred_lr

#Exporting required data into csv
#req_col = [0,14]
#cash_test.iloc[:,req_col].to_csv('cash_deposit_2017.csv',index=False)
# Code ends here

# Code starts here
#import sklearn.model_selection
from sklearn.ensemble import RandomForestRegressor

X_train = X_train_original
y_train = y_train_original
X_val = X_val_original
y_val = y_val_original

X_test = X_test_original

#using random forest regressor
rf_reg = RandomForestRegressor(n_estimators=300,random_state=0)
rf_reg.fit(X_train,y_train)

y_pred_rf1 = rf_reg.predict(X_val)
accuracy_rf = rf_reg.score(X_val,y_val)
print("\nAccuracy Random forest:", accuracy_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val,y_pred_rf1))
print("\nRandom Forest regression : Root Mean Square Error:",rmse_rf)

#Actual predictions
y_pred_rf = rf_reg.predict(X_test)
#print("y_pred_rf:", y_pred_rf)
Data_Auditor.Remove_Columns(cash_test,'deposit_amount_2017')

cash_test['deposit_amount_2017'] = y_pred_rf

#Exporting required data into csv
req_col = [0,14]
cash_test.iloc[:,req_col].to_csv('cash_deposit_2017.csv',index=False)
# Code ends here

# Code starts here
from sklearn.neighbors import KNeighborsRegressor

X_train = X_train_original
y_train = y_train_original
X_val = X_val_original
y_val = y_val_original

X_test = X_test_original

#using KNn Regression
kn_reg = KNeighborsRegressor(n_neighbors=5)
kn_reg.fit(X_train,y_train)

y_pred_kn1 = kn_reg.predict(X_val)
accuracy_kn = kn_reg.score(X_val,y_val)
print("\nAccuracy of KNN:", accuracy_kn)
rmse_kn = np.sqrt(mean_squared_error(y_val,y_pred_kn1))
print("\nKNeighbors regression : Root Mean Square Error:",rmse_kn)

y_pred_kn = kn_reg.predict(X_test) 
#print("y_pred_kn:",y_pred_kn)
Data_Auditor.Remove_Columns(cash_test,'deposit_amount_2017')

cash_test['deposit_amount_2017'] = y_pred_kn

#Exporting required data into csv
#req_col = [0,14]
#cash_test.iloc[:,req_col].to_csv('cash_deposit_2017.csv',index=False)
# Code ends here

# Code starts here
#Grid Search Cross Validation
#Pro Model -----> Ridge Model
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

X_train = X_train_original
y_train = y_train_original
X_val = X_val_original
y_val = y_val_original

X_test = X_test_original

lambdas=np.linspace(1,10,100)
params={'alpha':lambdas}
ridge_model = Ridge(fit_intercept=True)
grid_search = GridSearchCV(ridge_model,param_grid=params,cv=10,
                           scoring='neg_mean_absolute_error')
grid_search.fit(X_train,y_train)
grid_search.best_estimator_
grid_search.cv_results_
grid_search.best_estimator_.score(X_train,y_train) #R-Square

ridge_model_output_train = grid_search.best_estimator_.predict(X_train)
y_pred_cv1 = grid_search.best_estimator_.predict(X_val)
accuracy_cv = grid_search.best_estimator_.score(X_val,y_val)
print("\nAccuracy of Grid Search CV:", accuracy_cv)
rmse_cv = np.sqrt(mean_squared_error(y_val,y_pred_cv1))
print("\nGrid Search CV : Root Mean Square Error:",rmse_cv)

#Actual Prediction
ridge_model_output_train = grid_search.best_estimator_.predict(X_train)
y_pred_cv = grid_search.best_estimator_.predict(X_test)

cash_test['deposit_amount_2017'] = y_pred_cv

#Exporting required data into csv
#req_col = [0,14]
#cash_test.iloc[:,req_col].to_csv('cash_deposit_2017.csv',index=False)
# Code ends here

#Accuracy Comparision
accuracy = []
accuracy.append(accuracy_dt)
accuracy.append(accuracy_lr)
accuracy.append(accuracy_rf)
accuracy.append(accuracy_kn)
accuracy.append(accuracy_cv)

rmse = []
rmse.append(rmse_dt)
rmse.append(rmse_lr)
rmse.append(rmse_rf)
rmse.append(rmse_kn)
rmse.append(rmse_cv)

accuracydf = pd.DataFrame(list(zip(accuracy,rmse)), columns=['Accuracy','RMSE'])
accuracydf.set_index(pd.Index(['Decison Tree', 'Linear Regression', 'RandomForest Regressor', 'KNeighbors Regressor','GridSearch CV']),inplace=True)
print(accuracydf)

'''
Conclusion:
    i. Decison Tree: "rmse":  252.793706
    ii. Linear Regression: "rmse": 28097.925069
    iii. RandomForest Regressor: "rmse":  183.695896
    iv. KNeighbors Regressor: "rmse": 25056.872753
    v. GridSearch CV: "rmse": 206.468244
        
    Considering RMSE Value: Best suited model for prediction is:
        Random Forest Regressor having lowest value of RSME
'''
