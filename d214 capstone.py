# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 02:21:26 2024

@author: grade
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score
import scipy.stats as stats






customer_data = pd.read_excel('C:/Users/grade/Downloads/CAPSTONE FILES/csvs/customer_data.xlsx')
sales_data = pd.read_excel('C:/Users/grade/Downloads/CAPSTONE FILES/csvs/sales_data.xlsx')

df = pd.merge(customer_data,sales_data, on="customer_id")

print(df.shape)

print(df.duplicated().value_counts()) #check for duplcaite rows and columns, 99457 False, matching number of rows, indicating no duplicate rows

print(df.isnull().sum()) #119 null values are found on the age columns, they will be removed


df.dropna(inplace = True)

print(df.isnull().sum()) #verifying that no nulls in all columns
print(df.shape)  #verifying that there are 119 less rows now


#these two columns are merely transaction identifiers and arent customer characteristics, and as such will be removed
df.drop('customer_id', axis=1, inplace=True)
df.drop('invoice_no', axis=1, inplace=True)
df.drop('invoice date',axis=1, inplace=True)


print(df.info()) #verifying that the columns have been removed

#utlizing a boxplot to identify outliers in price columns, ie how much a customer has spent
# =============================================================================
plt.figure(1)
pricePlot = df.boxplot(column = ['price'])
# =============================================================================

#from the boxplot, values above 2500 are outliers
#since there are no values around 2750, this will be used as a cutoff

df.loc[df.price > 2750,'price'] = df['price'].median() #substitutes all values above 2750 with mean

# =============================================================================
plt.figure(2)
pricePlot = df.boxplot(column = ['price']) #new boxplot, values are much close together now
# 
plt.figure(3)
agePlot = df.boxplot(column = ['age']) #no outliers in age column
# 
plt.figure(4)
quantityPlot =  df.boxplot(column = ['quantity']) #no outliers in quantity column
# 
plt.figure(5)
plt.hist(df['price'],bins = 5) #from visual inspection, data is not normally distributed
# =============================================================================


#standardizing age and price columns into z-scores
scaler = StandardScaler()

scaler.fit(df[['age']])
df['age'] = scaler.transform(df[['age']])
scaler.fit(df[['price']])
df['price'] = scaler.transform(df[['price']])






#one hot encoding on categoricals except cities due to high dimensionailty
#https://www.geeksforgeeks.org/ml-one-hot-encoding/ assistance

one_hot_col = df.columns.to_list()
one_hot_col.remove('price')
one_hot_col.remove('shopping_mall')
one_hot_col.remove('age')
one_hot_col.remove('quantity')


encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[one_hot_col])

encoded_df = pd.DataFrame(encoded,columns = encoder.get_feature_names_out(one_hot_col))

print(df.shape)
print(encoded_df.shape)

df = pd.concat([df,encoded_df],axis = 1)
df = df.drop(one_hot_col,axis = 1)

df.dropna(inplace = True)






    


#frequency encoding for cities

#checking frequency of each city to ensure no same values
#https://letsdatascience.com/frequency-encoding/ discussion on frequency encoding, may utilize for discussion on it
print(df['shopping_mall'].value_counts())
df['shopping_mall'] = df['shopping_mall'].map(df['shopping_mall'].value_counts())







#values from value_counts mapped to corresponding column values

#no city appears at the same frequency
#converting each city to frequency at which the city appears in dataset


#creating elbow plot to identify the best K for the final model
distortions = []
clustNum = range(2,20)

for i in clustNum:
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(df)
    distortions.append(model.inertia_)
    
# =============================================================================
plt.figure(6)
sns.lineplot(x = clustNum,y = distortions)
# =============================================================================
#very sharp elbow point at k = 4, so 4 is optimal number of clusters for the model

model = KMeans(n_clusters = 4, init = 'k-means++',random_state = 1)
model.fit(df)

#number of customers in each cluster
#print(np.unique(model.fit_predict(df),return_counts=True))

clusters = model.fit_predict(df)

cluster0 = df[clusters == 0]
cluster1 = df[clusters == 1]
cluster2 = df[clusters == 2]
cluster3 = df[clusters == 3]

# =============================================================================
plt.figure(7)
plt.hist(cluster0['price'])
plt.xlabel("Price")
plt.ylabel("number of customers")
plt.title("Group1 Spending")
# 
plt.figure(8)
plt.hist(cluster1['price'])
plt.xlabel("Price")
plt.ylabel("number of customers")
plt.title("Group2 Spending")
# 
plt.figure(9)
plt.hist(cluster2['price'])
plt.xlabel("Price")
plt.ylabel("number of customers")
plt.title("Group3 Spending")
# 
plt.figure(10)
plt.hist(cluster3['price'])
plt.xlabel("Price")
plt.ylabel("number of customers")
plt.title("Group4 Spending")
# =============================================================================

print("Silhouette Score: "+str(silhouette_score(df,model.labels_)))

print("Cluster centroids:")
centroid_df = pd.DataFrame(model.cluster_centers_,columns = df.columns)
print(centroid_df)

#performing ANOVA on price

ANOVA = stats.f_oneway(cluster0.price,cluster1.price,cluster2.price,cluster3.price)
print(ANOVA)

#p-value > 0.05, the mean difference is not statistically significantly different




















