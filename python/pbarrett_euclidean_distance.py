# pbarrett_euclidean_distance.py
import numpy as np
import pandas as pd
import scipy.spatial.distance as sd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

df = pd.read_csv('data/table1.csv')
df.shape
# (3, 3)
df
        var1  var2
person            
1       20.0  80.0
2       30.0  44.0
3       90.0  40.0

# make the row entity the index so we can run scipy distance package
df.set_index('person', inplace=True)
df.shape
# (3,2)
df
df = df.astype(float)
df.dtypes

# calculate distance between each of three persons across two variables using scipy.spatial.distance
# requires arrays, not df columns
table1_ar = df.to_numpy()
#var1_ar = df['var1'].to_numpy()
#var2_ar = df['var2'].to_numpy()
sd.pdist(table1_ar)
# array([37.36308338, 80.62257748, 60.13318551])

# calculate the distance between the two variables over the sample of 3 persons:
sd.euclidean(var1_ar, var2_ar)
# 79.347

## Normalized Euclidean distance

df_table2 = pd.read_csv('data/table2.csv')
# standardize to unit mean and variance
df_table2.set_index('Variable', inplace=True)
Person1_ar = df_table2['Person1'].to_numpy()
Person2_ar = df_table2['Person2'].to_numpy()
sd.euclidean(Person1_ar, Person2_ar)
# 100.03

# normalized, double-scaled euclidean
# 
df_table4 = pd.read_csv('data/table4.csv')
df_table4.set_index('Variable', inplace=True)
 
# make the range (max-min) the denominator we use to scale the distance
md = ((df_table4.max(axis=0).to_numpy()) - (df_table4.min(axis=0).to_numpy()))**2
md.shape
# (2,)
md
# array([49.  , 32.49])

#rows and columns in data matrix (dm)
row_cnt = len(df_table4.index)
col_cnt = df_table4.shape[1]
# initialize distance matrix
dm = np.zeros((row_cnt, row_cnt))

for i in range(row_cnt):
    for j in range(row_cnt):
        dm[i,j] = np.sqrt(sum( ((df_table4.iloc[i,:] - df_table4.iloc[j,:])**2)**2 / md))
dm.shape
# (10,10)
np.savetxt('output/dm.csv', dm, delimiter=',')

# "double normalize" all entries in distance matrix by the sqrt of the number of variables
dm2 = dm*(1/np.sqrt(len(md)))
dm2.shape
# (10,10)
np.savetxt('output/dm_double_normed.csv', dm2, delimiter=',')

# this computes distance between each of the 10 vars
# let's transpose and see if we get same answer as barrett

df_table4_t = df_table4.transpose()
df_table4_t.shape
# (2,10)
# make the range (max-min) the denominator we use to scale the distance
md_t = ((df_table4_t.max(axis=0).to_numpy()) - (df_table4_t.min(axis=0).to_numpy()))**2
md_t.shape
# (10,0)
md_t
#rows and columns in data matrix (dm)
row_cnt = len(df_table4_t.index)
col_cnt = df_table4_t.shape[1]
# initialize distance matrix
dm_t = np.zeros((row_cnt, row_cnt))

for i in range(row_cnt):
    for j in range(row_cnt):
        dm_t[i,j] = np.sqrt(sum( ((df_table4_t.iloc[i,:] - df_table4_t.iloc[j,:])**2)**2 / md_t))
dm_t.shape
# (2,2)
np.savetxt('output/dm_t.csv', dm_t, delimiter=',')

# "double normalize" all entries in distance matrix by the sqrt of the number of variables
dm2_t = dm_t*(1/np.sqrt(len(md_t)))
dm2_t.shape
# (2,2)
np.savetxt('output/dm_t_double_normed.csv', dm2_t, delimiter=',')








# save as dataframe so we can add columns and reshape

df3 = pd.DataFrame(data=dm2, index = df2.index, columns = df2.index)
df3.head()
df3.shape
# put each pair in separate row
long_form = df3.unstack()
type(long_form)

len(long_form)
# rename columns and turn into a dataframe
long_form.index.rename(['Store_Number1', 'Store_Number2'],inplace=True)
long_form = long_form.to_frame('euclidean_distance_v2').reset_index()
# get rid of distance = 0 records
mask2 = long_form['euclidean_distance'] > 0
long_form2 = long_form.loc[mask2]
long_form2.shape
long_form2['Store_number1'].nunique()
# add region to each store variable so we can select stores NOT in the same region
temp_vars = ['Store_Number','Region']
df_temp = df_mask[temp_vars]
df_temp.shape
long-frm3 = long_form2_merge(df_temp, left_on='Store_Number1'. right_on='Store_number')
long_form3.shape
long_form3.rename(columns = {'Region: 'StoreNumber_1_Region'}, inplace=True)
long_form3.drop(['Store_number'], axis=1, inplace=True)
long_form3.shape
long_form4 = long_form3.merge(df_temp, left_on='StoreNumber2'}, inplace=True)
long_form4.rename(columns = {'Region': 'StoreNumber2_Region'}, inplace=True)
long_form4.drop(['Store_number'], axis=1, inplace_True, ignore_index=True)
long_form4.shape
# drop rows where stores are in same region as test and control stores must be indifferent regions
long_form5 = long_form4[['Store_number1_region'] != longform4['Store_Number2_region'])]
long_form5.shape
long_form6 = long_form5.sort_values(['Store_number1',euclidean_distance_v2'])
long_form6.shape
long_form6.to_csv('output/euclidean_dist_v2.csv')

long_form7 = long_form6[long_form6['row_num'] == 1)
long_form7.shape]
long_form7.to_csv('output/euclidean_dist_v2_min.csv')

df2.index

