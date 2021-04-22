# Importing libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Reading the dataset:

dataset = pd.read_csv("train-data.csv")



# cleaning the data:

dataset.isnull().sum() / len(dataset) * 100

# Considering mileage column:

dataset["Mileage"].value_counts()
dataset["Mileage"].isnull().sum()
dataset["Mileage"].unique()
        
  
dataset1 = dataset[dataset["Mileage"].notnull()]

dataset1.reset_index(inplace = True, drop = True)

m = dataset1["Mileage"].str.split(" ")
m.isnull().sum()

mileage_new = []

for i in m:
    if i[0] == np.nan:
        i[0] == 0.0
        mileage_new.append(i[0])
    else:
        mileage_new.append(i[0])

len(mileage_new)

mileage_new = pd.Series(mileage_new).astype(float)
mileage_new.isnull().sum()

mileage_new.count()
    
dataset1["Mileage"] = pd.Series(mileage_new)

dataset1["Mileage"].isnull().sum()

# Treating the engine column:
    
dataset1["Engine"].isnull().sum()
dataset1["Engine"].value_counts()
dataset1["Engine"].unique()

dataset1 = dataset1[(dataset1["Engine"].notnull())]

dataset1.reset_index(drop = True, inplace = True)

engine_new = []

e = dataset1["Engine"].str.split(" ")

for i in e:
    engine_new.append(i[0])
    
len(engine_new)

engine_new = pd.Series(engine_new).astype(int)

engine_new.isnull().sum()

dataset1["Engine"] = engine_new
dataset1["Engine"].isnull().sum()

# Treating power column:
    
dataset1["Power"].isnull().sum()
dataset1["Power"].value_counts()
dataset1["Power"].nunique()

power_new = []

p = dataset1["Power"].str.split(" ")

for i in p:
    power_new.append(i[0])
    
len(power_new)

power_new = pd.Series(power_new)
power_new.isnull().sum()

    
dataset1["Power"] = power_new

dataset1["Power"].isnull().sum()

dataset1[(dataset1["Power"] == "null")]

dataset1["Power"] = dataset1["Power"].replace("null", np.nan)
dataset1["Power"] = dataset1["Power"].astype(float)

dataset1["Power"].dtype

# There are 107 null values in Power column. We will replace them either with
    # mean or median, basis the distribution of this feature.

# Checking the distribution:
    
sns.distplot(dataset1["Power"])
plt.show()

# Since it can be seen that the distribution is right skewed, we will replace
    # the null value with median.
    
dataset1["Power"].fillna(dataset1["Power"].median(),inplace = True)

dataset1["Power"].isnull().sum()


# Treating the new_price column:
    
dataset1["New_Price"].isnull().sum()/len(dataset1) * 100

dataset1["New_Price"].dtypes

# It can be seen that since there are 86.23% null values in the new_price 
    # column, we will be dropping this column.
    
dataset1.drop(labels = ["New_Price"], axis = 1, inplace = True)
    
# Treating the seats column:
    
dataset1["Seats"].isnull().sum()
dataset1["Seats"].value_counts()
dataset1["Seats"].unique()    

dataset1 = dataset1[dataset1["Seats"].notnull()]

dataset1.reset_index(drop = True, inplace = True)

dataset1 = dataset1[dataset1["Seats"] != 0]

dataset1.reset_index(drop = True, inplace = True)

dataset1["Seats"].unique()

dataset1["Seats"] = dataset1["Seats"].astype("O")

# Dropping the Unnamed:0, i.e., another index column:
    
dataset1 = dataset1.iloc[:,1:]

# Treating the name column:
    
dataset1["Name"].isnull().sum()
dataset1["Name"].unique()
dataset1["Name"].nunique()

# Treating location column:
    
dataset1["Location"].isnull().sum()
dataset1["Location"].unique()

#  Treating Year column:
    
dataset1["Year"].isnull().sum()
dataset1["Year"].unique()
dataset1["Year"] = dataset1["Year"].astype("O")

# Treating Kilometers_Driven column:
    
dataset1["Kilometers_Driven"].isnull().sum()
dataset1["Kilometers_Driven"].unique()
dataset1["Kilometers_Driven"].dtypes

# Treating Fuel_Type column:
    
dataset1["Fuel_Type"].isnull().sum()
dataset1["Fuel_Type"].unique()
dataset1["Fuel_Type"].value_counts(normalize = True)

# Treating Transmission column:
    
dataset1["Transmission"].isnull().sum()
dataset1["Transmission"].unique()
dataset1["Transmission"].value_counts(normalize = True)

# Treating Owner_Type column:
    
dataset1["Owner_Type"].isnull().sum()
dataset1["Owner_Type"].unique()
dataset1["Owner_Type"].value_counts(normalize = True)


# Treating Price column:
    
dataset1["Price"].isnull().sum()
dataset1["Price"].unique()


###############################################################################
# Till now, we have done the basic cleaning in the data. Now it can be used for
    # EDA and modelling.
###############################################################################


############################### EDA AND STATS #################################
    
# Before EDA and stats, we will be divding the columns into numerical and 
    # categorical:
        
    
dataset1_int = dataset1.iloc[:,[3,7,8,9,11]]
dataset1_str = dataset1.iloc[:,[1,2,4,5,6,10]]

sns.distplot(dataset1_int["Kilometers_Driven"])
plt.show()

sns.distplot(dataset1_int["Mileage"])
plt.show()


###############################################################################

########################### Preparing Base Model ##############################


# Dummy Variable Encoding:
    
dataset1_str = pd.get_dummies(dataset1_str, drop_first = True)

y = dataset1_int.iloc[:,-1].values

dataset1_int = dataset1_int.drop("Price", axis = 1)


# Preparing Final Dataset:
    
final_dataset1 = pd.concat([dataset1_int, dataset1_str], axis = 1)

X = final_dataset1.values


# Splitting the dataset into training and test set:
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)    

# Developing the Base Model:
    
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Validating base model:
    
lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)


# It can be seen that there neither overfitting nor underfitting present in the 
    # model
    
###############################################################################
################## Improving the Base Model Step by Step #######################

dataset1.dtypes

import statsmodels.api as sm

model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()

# Identifiying the significant and insignificant features in the model>

print(result.summary())


# Basis the summary, the p-values of the continuous features are analyzed.

# It can be seen that the feature - "Engine" has a p-value far bigger than 
    # 0.05, hence it means that it is an insignificant feature the model.
    
# Hence we will drop "Engine" feature.

###############################################################################
################### Updated Base Model Basis P-Values #########################

final_dataset1_update_1 = pd.concat([dataset1_int.iloc[:,[0,1,3]], dataset1_str], axis = 1)

X = final_dataset1_update_1.values


# Splitting the dataset into training and test set:
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)    


# Developing the Base Model:
    
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Validating base model:
    
lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)


import statsmodels.api as sm

model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()

# Identifiying the significant and insignificant features in the model>

print(result.summary())

# After removing the "Engine" feature, it can be seen that the value of R-square
    # inceresed from 71.9 to 72.2. This shows improvement in the model.
    
    

# Checking how good the model is:
    
y_pred = result.predict(sm.add_constant(X_test))

from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

# r2 is 0.76, i.e. more than 0.5. Hence it is a good model. We will try to 
    # impove it more by applying regularization

########################### Applying Regularization ###########################

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)

print(ridge.score(X_train, y_train))
print(ridge.score(X_test, y_test))

# It can be seen that with ridge, there is no significant imporvement in the 
    # model.

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)

print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))

# Lasso has reduced the score of the model significantly, hence not useful.

from sklearn.linear_model import ElasticNet
elastic = ElasticNet()
elastic.fit(X_train, y_train)

print(elastic.score(X_train, y_train))
print(elastic.score(X_test, y_test))

# Even ElasticNet has not improved the model hence not useful.

# Applying K-Fold Cross Validation:
    
from sklearn.model_selection import cross_val_score
cross_val_score(ridge, X_train, y_train)
cross_val_score(lasso, X_train, y_train)
cross_val_score(elastic, X_train, y_train)

# Basis ridge cross val score, it can be seen that sample bias is present in 
    # the data.

###############################################################################

#Basis the above models, it can be seen that we can either use LinearRegression
    # or Ridge Regression.
    
############################### Model Tuning ##################################

ridge_params = {'alpha' : np.arange(0, 11, 0.1)}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(ridge, ridge_params)
grid.fit(X_train, y_train)

grid.best_params_

# 0.1 is the best value of alpha here.

grid.best_score_
best_ridge = grid.best_estimator_

best_ridge.score(X_test, y_test)

y_pred_ridge = best_ridge.predict(X_test)

mean_squared_error(y_test, y_pred_ridge)
r2_score(y_test, y_pred_ridge)

# there is no change in the r2_score of ridge and tuned ridge.

lasso_params = {'alpha' : [0,11,0.1]}

from sklearn.model_selection import GridSearchCV
grid1 = GridSearchCV(lasso, lasso_params)
grid1.fit(X_train, y_train)

grid1.best_params_
grid1.best_score_
best_lasso = grid1.best_estimator_

y_pred_lasso = best_lasso.predict(X_test)

r2_score(y_test, y_pred_lasso)

# the r2_score of tuned lasso(76.64) is wayy better than the normal 
    # lasso(61.16) and is at par with ridge and tuned ridge.
    
elastic_params = {'alpha' : [0,11,0.1]}

from sklearn.model_selection import GridSearchCV
grid3 = GridSearchCV(elastic, elastic_params)
grid3.fit(X_train, y_train)

grid3.best_params_
grid3.best_score_
best_elastic = grid3.best_estimator_

y_pred_elastic = best_elastic.predict(X_test)

r2_score(y_test, y_pred_elastic)

# the r2_score of tuned elasticnet(76.64) is wayy better than the normal 
    # elastic(61.17) and is at par with ridge and tuned ridge.
    


############################### Feature Engineering ###########################


dataset1.dtypes

# List of features in data:
    

    # Location              Dummy
    # Engine                Log_trans
    # Year                  Dummy
    # Kilometers_Driven     Log_trans
    # Fuel_Type             Dummy
    # Transmission          Dummy
    # Owner_Type            Dummy / LE
    # Mileage               Log_trans
    # Power                 Log_trans
    # Seats                 Dummy

    
dataset1_int.hist(bins = 100)


plt.hist(dataset1_int["Kilometers_Driven"])
plt.show()

dataset1_int["Kilometers_Driven"].value_counts()
dataset1_int["Kilometers_Driven"].nunique()

sns.scatterplot(dataset1["Kilometers_Driven"], dataset1["Price"])
plt.show()

sns.boxplot(dataset1["Mileage"])
plt.show()

sns.boxplot(dataset1["Power"])
plt.show()

dataset1["Location"].value_counts()
dataset1["Location"].nunique()

stats.chi2_contingency(pd.crosstab(dataset1["Location"], dataset1["Year"]))
stats.chi2_contingency(pd.crosstab(dataset1["Location"], dataset1["Fuel_Type"]))
stats.chi2_contingency(pd.crosstab(dataset1["Location"], dataset1["Transmission"]))
stats.chi2_contingency(pd.crosstab(dataset1["Location"], dataset1["Owner_Type"]))
stats.chi2_contingency(pd.crosstab(dataset1["Location"], dataset1["Seats"])) ##

dataset1["Year"].value_counts()
dataset1["Year"].nunique()

stats.chi2_contingency(pd.crosstab(dataset1["Year"], dataset1["Fuel_Type"]))
stats.chi2_contingency(pd.crosstab(dataset1["Year"], dataset1["Transmission"]))
stats.chi2_contingency(pd.crosstab(dataset1["Year"], dataset1["Owner_Type"]))
stats.chi2_contingency(pd.crosstab(dataset1["Year"], dataset1["Seats"]))

dataset1["Fuel_Type"].value_counts()

stats.chi2_contingency(pd.crosstab(dataset1["Fuel_Type"], dataset1["Transmission"]))
stats.chi2_contingency(pd.crosstab(dataset1["Fuel_Type"], dataset1["Owner_Type"])) ##
stats.chi2_contingency(pd.crosstab(dataset1["Fuel_Type"], dataset1["Seats"]))


dataset1["Transmission"].value_counts()

stats.chi2_contingency(pd.crosstab(dataset1["Transmission"], dataset1["Owner_Type"])) ##
stats.chi2_contingency(pd.crosstab(dataset1["Transmission"], dataset1["Seats"]))


dataset1["Owner_Type"].value_counts()

stats.chi2_contingency(pd.crosstab(dataset1["Owner_Type"], dataset1["Seats"]))

dataset1["Seats"].value_counts()
dataset1["Seats"].nunique()


###############################################################################

dataset_finally = pd.concat([dataset1_int.iloc[:, [1,3]], dataset1_str, dataset1.iloc[:,-1]], axis = 1)

final_dataset1["Engine"] = np.log1p(final_dataset1["Engine"])
final_dataset1["Kilometers_Driven"] = np.log1p(final_dataset1["Kilometers_Driven"])
final_dataset1["Mileage"] = np.log1p(final_dataset1["Mileage"])
final_dataset1["Power"] = np.log1p(final_dataset1["Power"])

trial = pd.concat([dataset_finally.iloc[:,:2],dataset_finally.iloc[:,12:37], dataset_finally.iloc[:,40:-1]], axis = 1)

X = final_dataset1.values
y = dataset1_int.iloc[:,-1].values

# Splitting the dataset into training and test set:
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)    

# Developing the Base Model:
    
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Validating base model:
    
lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)


import statsmodels.api as sm

model = sm.OLS(y_train, sm.add_constant(X_train))
result = model.fit()

# Identifiying the significant and insignificant features in the model>

print(result.summary())


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)

print(ridge.score(X_train, y_train))
print(ridge.score(X_test, y_test))
