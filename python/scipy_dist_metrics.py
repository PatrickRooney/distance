# scipy distance metrics

import numpy as np
import pandas as pd
import scipy as scipy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

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


pd.options.display.float_format="{:2f}".format
scipy.spatial.distance.pdist(arr)
#array([13397.63781418,  7516.15320493,  7759.16155264,  7422.74922114,
#       10555.18772926, 11788.35734952, 18116.7220269 ,  5340.65941621,
#        9322.08458447,  9163.07601191])

scipy.spatial.distance.pdist(arr, 'sqeuclidean')
#array([1.79496699e+08, 5.64925590e+07, 6.02045880e+07, 5.50972060e+07,
#       1.11411988e+08, 1.38965369e+08, 3.28215617e+08, 2.85226430e+07,
#       8.69012610e+07, 8.39619620e+07])

scipy.spatial.distance,pdist(arr, 'cosine')
# array([0.28691865, 0.17287879, 0.28873409, 0.22587285, 0.16352266,
#       0.12320638, 0.64443329, 0.03558186, 0.27484049, 0.39737309]))
