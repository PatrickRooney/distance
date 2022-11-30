# Euclidean double-scaled distance
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

df = pd.read('../data/tabl1.csv')
df.shape
# make the row entity the index so we can run scipy distance package
df.set_index('person', inplace=True)
df.shape
df = df.astype(float)
df.dtypes

# make the range (max-min) the denominator we use to scale the distance
md = ((df.max(axis=0).to_numy()) - (df.min(axis=0).to_numpy()))**2
md.shape
md
#rows and columns in data matrix (dm)
row_cnt = len(df.index)
col_cnt = df.shape[1]
# initialize distance matrix
dm = np/zeros((row_cnt, row_cnt))

for i in range(row_cnt):
    for j in range(row_cnt):
        dm[i,j] = np.sqrt(sum( ((df.iloc[i,:] - df.iloc[j,:])**2)**2 / md))
dm.shape
np.savetxt('../output/dm.csv', dm, delimiter=',')

# "double normalize" all entries in distance matrix by the sqrt of the number of variables
dm2 = dm*(1/np.sqrt(len(md)))
dm2.shape
np.savetxt('../output/dm.csv', dm, delimiter=',')

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

