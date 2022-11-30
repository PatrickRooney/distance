# get sample of 100 records from Wholesale_customers_data.csv

import pandas as pd
df = pd.read_csv('data/Wholesale_customers_data.csv')
df_samp = df.sample(n=5)
df_samp.to_csv('data/Wholesale_customers_samp.csv', index=False)