


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msn
import math
from sklearn.metrics import r2_score
data=pd.read_csv('/content/iris.data',names=['sepal_length','sepal_width','petal_lenth','petal_width','flower'])

data.head(5)

independent_value=data.iloc[ : , :-1]
dependent_value=data.iloc[ : ,4]
dep=dependent_value

msn.matrix(data)

from sklearn.preprocessing import LabelEncoder as label
label_of_dependent_values=label()
dependent_value=label_of_dependent_values.fit_transform(dependent_value)
Iris_setosa=0
Iris_versicolor=1
Iris_virginica=2
new_dep_df=pd.DataFrame(dep,dependent_value)


print(dependent_value.ndim)

"""## Linear Regression

To find beta values
"""

def linear_regression(n,y):
  X_values=n.drop(labels='flower',axis=1) 
  X_values.insert(loc=0,column='B',value=1) # adding the Y intersept (Bo)
  X_values.to_numpy()
  data_transpose=np.transpose(X_values)
  Xx=data_transpose @ X_values         
  Xy=np.dot(data_transpose,y)   
  inverse_Xx=np.linalg.inv(Xx)
  beta_values=inverse_Xx@Xy
  print('beta variables are',beta_values) # betas values are found
  
  x_test=n
  y_test=y
  y_predict=[]
  for i , r in n.iterrows():
     sum=beta_values[0]+(beta_values[1]*r['sepal_length'])+(beta_values[2]*r['sepal_width'])+(beta_values[3]*r['petal_lenth'])+(beta_values[4]*r['petal_width'])
     y_predict.append(sum) 
     
  return r2_score(y_test,y_pred)
a=linear_regression(data,dependent_value)
print('performance score is',a)

"""### **Classification**"""

k=3
dist=[]
for i ,r  in independent_value.iterrows():
  sum_of_the_x=r['sepal_length']**2+r['sepal_width']**2+r['petal_lenth']**2+r['petal_width']**2
  Eucledean_dist=math.sqrt(sum_of_the_x)
  dist.append(Eucledean_dist)
df_of_flower_and_dist=pd.DataFrame(data=[dependent_value,dist])
new_df_of_flower_and_dist=df_of_flower_and_dist.T
df_sort=new_df_of_flower_and_dist.sort_values(1)
df_check=df_sort.head(3)
flow_num_list=[]
for i in df_check[0]:
   aj=i
   flow_num_list.append(i)
ele_1=flow_num_list[0]
ele_2=flow_num_list[1]
ele_3=flow_num_list[2]
if ele_1==ele_2==ele_3:
  out1=ele_1
elif ele_1==ele_2 and ele_1!=ele_3:
  out1=ele_1
else:
  out1=ele_2
if out1==Iris_setosa:
  print('the classification result is , Iris_setosa')
elif out1==Iris_versicolor:
  print('the classification result is , Iris_Iris_versicolor')
else:
  print('the classification result is , Iris_Iris_versicolor')

"""### Cross Validation"""

