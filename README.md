# Euclidean Double Scaled Distance README

1. Title: Euclidean Double Scaled Distance
2. Project Description

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


 3. How to Install and Run the Project:

I created the [distance repo](https://github.com/PatrickRooney/distance) on a laptop running Ubuntu version 20.04.5 and Python version 3.8.10.

After cloning the [distance repo](https://github.com/PatrickRooney/distance) to your local directory, you will need to do the following to run the script using the example "toy" dataset:

Check that the values in the `config.yml` file have the correct pathnames.  Change them to appropriate values if you are not using the example dataset.

Activate the virtual environment
    `./source venv/bin/activate`

After filling the applicable values in the config.yml file, the script may be run at the Linux command line by issuing

```
python3 filepath/euclidean_double_scale_distance.py /.config.yml`
```
for example, I have a virtual environment where python3 is stored in `./venv/bin` and the python script is stored in `./python`:

```
./venv/bin/python3 ./python/euclideandoublescaledstoredistances.py /.config.yml
```

(note you may still get the warning "A value is trying to be set on a copy of a slice from a DataFrame" but it produces correct output)
    
 4.  Directory Structure
 
 ```

├── data
├── docs
├── output
├── python
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── lib64 -> lib
    └── share
```

 
 
 
    How to Use the Project. ...
    Include Credits. ...
    Add a License. ...
    Badges.