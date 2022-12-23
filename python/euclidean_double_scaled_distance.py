# Euclidean double-scaled distance.py
'''
Script to compute "double-scaled" Euclidean distance between units (e.g., zip codes, regions, DMA's, stores) across multiple attributes (e.g., sales, income, clicks).  The distance measure is from P. Barett, 2005, "Euclidean Distance: raw, normalized, and double-scaled coefficients" found at https://www.pbarrett.net/techpapers.html.  The Euclidean distance is scaled by the range of values for each attribute and by the number of attributes.  This enables the distance measure to be computed with reference to the largest possible range in the data across the attributes.  The distance measure can range from 0 to 1.  The similarity of any unit with another may be easily obtained by the formula similarity = 1 - distance.

The script outputs a n x n distance matrix (where n is the number of units, or rows) as well as a table of all unit pairs with their distance/similarity.  This can be used to design tests where random assignment is not possible with test and control units that are as similar as possible.  This is not intended for matching relatively large number of units (1,000+ units), such as customers,  where other matching methods may be more efficient.

We assume input data is a rectangular matrix with rows representing the units (e.g., stores, regions, DMAs) and the columns (e.g., clicks, sales, income) representing attributes/characteristics of the units over which the distances/similarities will be measured.  We require the columns to be numeric and the file format to be comma-delimited (csv).

File paths and column names are stored in config.yml, so that the code does not need to be edited; just change the file paths/attributes in the config.yml file to read a new file or use a different set of attributes.

The config.yml file contains the following:
    DATA_PATH: './data/'
    OUTPUT_PATH: './output/'
    INPUT_FILE: 'Wholesale_customers_samp.csv'
    ROW_ENTITIES: 'Store'
    ID_COL: 
    DIST_COLS: ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
    DIMENSION_COLS: ['Channel', 'Region'] 
    OUTPUT_DIST: 'distance_matrix.csv'
    OUTPUT_PAIRS: 'distance_pairs.csv'

'''
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import yaml
import click

@click.command()
@click.argument("config_path")
def run(config_path='./config.yml'):
    config_path='config.yml'
    # Read in yaml file
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    read_path = config['DATA_PATH'] + config['INPUT_FILE']
    output_dist = config['OUTPUT_PATH'] + config['OUTPUT_DIST']
    output_pairs = config['OUTPUT_PATH'] + config['OUTPUT_PAIRS']
    # create pandas dataframe which may contain non-float columns
    df = pd.read_csv(read_path)
    # create numpy array containing only float columns
    df_subset = df[config['DIST_COLS']]
    df_subset = df_subset.astype(float)
    arr = df_subset.to_numpy()

    # make the range (max-min) the denominator we use to scale the distance
    md = (np.max(arr, axis=0) - np.min(arr,axis=0))**2
   
    #rows and columns in data matrix (dm)
    row_cnt = len(arr)
    col_cnt = arr.shape[1]
    # initialize distance matrix
    dm = np.zeros((row_cnt, row_cnt))

    for i in range(row_cnt):
        for j in range(row_cnt):
            dm[i,j] = np.sqrt(sum( (arr[i,:] - arr[j,:])**2 / md))
            #dm[i,j] = np.sqrt(sum( ((df.iloc[i,:] - df.iloc[j,:])**2)**2 / md))

    # "double normalize" all entries in distance matrix by the sqrt of the number of variables
    dm2 = dm*(1/np.sqrt(len(md)))
    
    # output distance matrix as pandas dataframe so we can label the rows and columns

    # if there is no ID column, use the index of the original dataframe to be the row
    # and column labels of the distance matrix
    # TODO: if there is a ID column, then use this as the row and columns of the distance matrix
    
    row_index_str = [str(x) for x in df.index.tolist()]
    row_names = [config['ROW_ENTITIES'] +'_' + s for s in row_index_str]
    col_names = row_names
    df_dist_matrix = pd.DataFrame(dm2, columns = col_names, index = row_names)
    df_dist_matrix.to_csv(output_dist, index_label = 'Row_Name', float_format = '%.2f')

    # reshape output distance matrix, such that each pair of units (e.g., stores) are on a separate row
    # with their distance and similarity (where similarity = 1 - distance)
    # this will make it easier to present in Flask or other tools and to be manipulated by the user
    
    pairs = df_dist_matrix.unstack()
    pairs.index.rename(['Pair_Member_1', 'Pair_Member_2'], inplace=True)
    df_pairs = pairs.to_frame().reset_index()
    df_pairs = df_pairs.rename(columns={0: 'distance'})
    df_pairs['similarity'] = 1 - df_pairs['distance']

    # get rid of the rows where an entity is paired with itself and distance is necessarily = 0
    df_pairs = df_pairs[df_pairs['Pair_Member_1'] != df_pairs['Pair_Member_2']]

    # logging pairs_form2.shape
    # logging pairs_form2['Store_number1'].nunique()

    # add dimensions (e.g., channel or region) to each row to enable reporting or for selecting units NOT in the same region (e.g., where test stimuli can only be applied to all stores within a region)
    # need to deal with variable number of dimension cols

    df_dimensions = df[config['DIMENSION_COLS']]
    # logging: df_temp.shape
    # add entity_id column to join to df_pairs
    df_dimensions.reset_index(names=['temp'], inplace=True)
    df_dimensions = df_dimensions.copy()
    df_dimensions['entity_id'] = 'Store_' + df_dimensions['temp'].apply(str)
    df_dimensions = df_dimensions.drop('temp', axis=1)
            
    df_pairs = df_pairs.merge(df_dimensions, left_on='Pair_Member_1', right_on='entity_id')
    # logging: pairs_form3.shape
    # rename dimension columns so it's clear they only apply to Pair_Member_1
    for d in config['DIMENSION_COLS']:
        df_pairs.rename(columns = {d: 'Pair_Member_1' + '_' + d}, inplace=True)
    
    df_pairs.drop(['entity_id'], axis=1, inplace=True)
    
    # logging: df_pairs.shape

    df_pairs = df_pairs.merge(df_dimensions, left_on='Pair_Member_2', right_on='entity_id')
    # logging: pairs_form3.shape
    # rename dimension columns so it's clear they only apply to Pair_Member_2
    for d in config['DIMENSION_COLS']:
        df_pairs.rename(columns = {d: 'Pair_Member_2' + '_' + d}, inplace=True)
    
    df_pairs.drop(['entity_id'], axis=1, inplace=True)

    # logging: df_pairs.shape
    # logging: df_pairs.columns

    df_pairs.sort_values(by=['Pair_Member_1', 'similarity'], ascending=False, inplace=True)
    
    df_pairs.to_csv(output_pairs, float_format = '%.2f', index=False)

if __name__ == "__main__":
    run()    


