"""
This file is for providing functions to extract relevant features from the data.

Dealing with all datatypes:
    dropping irrelevant features
Dealing with numerics:
    change zero values by:
        dropping rows with zeros in it
        replacing zeros with median
    scaling numeric values like:
        log scale
        box cox scale
        absolute value negative values
    binning values
    removing outliers with Z-score threshold
    standard scale numeric values

Dealing with datetime:
    convert datetime to days from first date

Dealing with categories:
    replace NaN with the most common category
    reduce dimensions of categorical data
    convert categories into one-hot encoding

=======================================================================

Usage Examples:

>>> train_X, valid_X, train_y, valid_y = fe.get_data(
    '../data/train.csv', '../data/train_labels.csv')
>>> trans = fe.DataCleaning()
>>> transformed_train_X = trans.transform(train_X)
>>> np_train_X = fe.transform_df(transformed_train_X)

=======================================================================

Feature Engineering Descriptions:

Numeric:
Binarization: turn data into binary data.
Weight Changes: instead of linear weights, we can have polynomial weights.
Binning: 
    put numeric values into bins, especially for inputs with outliers or large ranges
    fixed binning: decide on the bin sizes myself
    adaptive binning (usually safer): can divide bins into quartiles 
Numeric Transformations: 
    log: for skewed distributions as it tends to expand the values that fall 
         in the range of lower magnitudes and compress the range of higher magnitudes
    box cox: used when variabity in different regions are largely different
    arcsin: for porportions
Outlier Removals:
    z-score: parametric, usually 2.5, 3, 3.5
    Dbscan:  https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561
    Isolation Forest:
Dealing With Irregular Values:
    replace by mean or median: problem is that it doesn't account for uncertainty 
        in imputations and doesn't factor in correlations between features
feature combination:
    is there a way to combine features to make more insight? I don't see any reason right now.

Categorical:
If there's order, we can convert the categories into integers.
If there's no order, use one hot encoding.
If there's too many labels, group them into bins or use dimension reduction.
If there's irregular labels:
    replace by most frequent: problem is that it introduces biases
If there's missing values:
    replace by most common.
    Note that this is very tricky, and I really need to compare before and afters
        for training to see if this works.
"""


import json
import pprint
import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import scipy.stats as ss
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import condense_csv


#======================================================================
# Get the data.
#======================================================================


def convert_num_labels_to_cat(df, categories):
    """
    Some columns have integer values, but do not denote any order, 
    so we need to convert them into categories.

    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :param categories: list of column names to convert
    :type  categories: list
    """

    for cat in categories:
        df[cat] = df[cat].astype('category')

    return df


def get_data(data_path, label_path, valid_size=0.2):
    """
    Get the data, and split it into train and validation.
    Also, convert appropriate numeric labels to categorical dtypes.

    :param data_path: path of the data input
    :type  data_path: str
    :param label_path: path of the data label
    :type  label_path: str
    :param test_size: Proportion to split on for the validation size
    :type  test_size: float
    """

    from sklearn.model_selection import train_test_split

    data = pd.read_csv(data_path)
    labels = pd.read_csv(label_path)
    data = condense_csv.compress_X(data)
    labels = condense_csv.compress_labels(labels)

    X_train, X_valid, y_train, y_valid = train_test_split(
        data,
        labels,
        test_size=valid_size,
        random_state=42
    )

    X_train = convert_num_labels_to_cat(
        X_train, ['region_code', 'district_code'])
    
    X_valid = convert_num_labels_to_cat(
        X_valid, ['region_code', 'district_code'])
    
    X_train['status_group'] = y_train['status_group']
    X_valid['status_group'] = y_valid['status_group']
    
    return X_train, X_valid, y_train, y_valid


#======================================================================
# Create a list of classes based on json file.
#======================================================================


def get_starting_info():
    """
    Get the initial starting information from starting_info.json.

    :returns: a dictionary where the key is the column name and 
              the value is a list, where the first item of that list
              is a list of operations to perform on the column 
              and the second item is a string that justifies why 
              a certain operation is performed.
    :rtype:   dict
    """

    file = open('../preprocess/starting_info.json')
    starting_info = json.load(file)
    file.close()
    return starting_info


class ColumnOperations:
    """
    A class for representing column operations to be performed for the dataframe
    """
    
    def __init__(self, name, operations, justifications):
        """
        :param name: column name
        :type  name: str
        :param operations: a sequence of operations to be performed on the column
        :type  operations: list
        :param justification: justification for operation
        :type  justification: str
        """
        
        self.name = name
        self.operations = operations
        self.justifications = justifications


def create_col_instances():
    """
    Create a list of instances of the column operation class

    :param col_dict: dictionary where keys are column names and values are the class attributes
    :type  col_dict: dict
    :returns: list of class instances
    :rtype:   list
    """

    starting_info = get_starting_info()

    col_instances = []
    for col_name, attributes in starting_info.items():
        col_instances.append(ColumnOperations(
            col_name, attributes[0], attributes[1]))
    return col_instances

list_of_classes = create_col_instances()


#======================================================================
# Perform operations on the dataframe.
#======================================================================


def perform_operations(df, col_name, operations):
    """
    Execute operations on a certain column in the dataframe.
        Dtypes                  Operations:      Description:
        ALL                     drop             drop the entire column

        Numerics
                                log              perform log transformation on the column
                                box cox          perform box cox transformation on the column
                                drop0            drop all values with zeros in it
                                absneg           absolute value the negatives
                                median0          replace 0 with the median
                                binning_NUM      create NUM amount of bins
                                outlierZ_NUM     remove outliers with z score > NUM
                                shiftmin         subtract the columns by the minimum value

        Datetime
                                finddays         convert datetime to days since the first day

        Categorical/Object
                                cbinning_NUM     create bins where each bin must have occurences 
                                                 of NUM or higher
                                mostcommon       replace nan with most common category

    :param df: dataframe 
    :type  df: pandas.core.frame.DataFrame
    :param col_name: name of column
    :type  col_name: str
    :param operations: list of operations to perform on the certain column
    :type  operations: list
    :returns: transformed dataframe 
    :rtype:   pandas.core.frame.DataFrame
    """

    col = df[col_name]

    # iterate throughout the list of transformations for each column
    for operation in operations:
        if operation == 'drop':
            # immediately returns the dataframe since no more operations can be
            # performed on a drop column
            return df.drop(col_name, axis=1)

        # numeric columns
        elif str(col.dtype) in {'int8', 'int16', 'int32', 'float64'}:
            if operation == 'log':
                col = np.log(1 + col)  # to make sure no divide by zero
            elif operation == 'box cox':
                col = ss.boxcox(col + 0.001)  # to make sure no divide by zero
            elif operation == "drop0":
                df = df[col != 0]
                col = col[col != 0]
            elif operation == "absneg":
                col = col.abs()
            elif operation == "median0":
                from sklearn.preprocessing import Imputer
                col[col == 0] = np.nan
                imputer = Imputer(strategy="median")
                col = imputer.fit_transform(col.values.reshape(-1, 1))
            elif operation.split('_')[0] == 'binning':
                # name would be binning_NUM
                num = int(operation.split('_')[1])
                quantile_list = [i / (num - 1) for i in range(num)]
                # this column with DROP_ will eventually be dropped.
                # It is here if one were to reference the the bins
                df["DROP_" + col_name] = pd.qcut(
                    col,
                    q=quantile_list,
                    duplicates='raise',
                )
                col = pd.qcut(
                    col,
                    q=quantile_list,
                    duplicates='raise',
                    labels=quantile_list[1:]
                )
            elif operation.split('_')[0] == 'outlierZ':
                z = np.abs(ss.zscore(col))
                keep_values = z < float(operation.split('_')[1])
                df = df[keep_values]
                col = col[keep_values]
            elif operation == "shiftmin":
                col = col - col.min()
            else:
                raise ValueError('Not an available operation for numerics')

        # datetime columns
        elif str(col.dtype) in {'datetime64[ns]'}:
            # TODO: add more datetime dtypes (not sure if that is the only one)
            if operation == "finddays":
                # TODO: should NOT be min, will not generalize to validation/test
                col = (col - min(col)).dt.days

        # categorical or object columns
        elif str(col.dtype) in {'category', 'object'}:
            if operation.split('_')[0] == "cbinning":
                num = float(operation.split('_')[1])
                value_counts = col.value_counts()
                x = col.replace(value_counts)
                df[col_name][df[col_name] == '0'] = np.nan
                df[col_name] = df[col_name].cat.add_categories(['OTHER'])
                df[col_name] = df[col_name].fillna('OTHER')
                df.loc[x < num, col_name] = 'OTHER'
                return df
            elif operation == "mostcommon":
                most_common = col.value_counts().index[0]
                col = col.fillna(most_common)
            else:
                raise ValueError(
                    'Not an available operation for categoricals or objects')

        else:
            raise ValueError('Not an available data type')

    df[col_name] = col

    return df


def perform_operations_with_classes(df, list_of_instances):
    """
    For each instance of a class in a list,
    perform the operation(s) indicated in the instance.
    For example, one instance contains the name of the column and 
    list of operations to perform on that column.
    For each matching column in the dataframe, perform the operations.

    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :param list_of_instances: list of instances of a class
    :type  list_of_instances: list
    :returns: dataframe
    :rtype:   pandas.core.frame.DataFrame
    """

    for col_class in list_of_instances:
        df = perform_operations(df, col_class.name, col_class.operations)
    return df


def drop_col_with_DROP(df):
    """
    Drop columns that have a name of DROP_XXX.
    
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :returns: dataframe of dropped columns
    :rtype:   pandas.core.frame.DataFrame
    """
    
    for col in df.columns:
        if col.split('_')[0] == 'DROP':
            df = df.drop([col], axis=1)
    return df


class DataCleaning(BaseEstimator, TransformerMixin):
    """
    Class for performing data cleaning, so the same can be applied to the test case
    """

    def __init__(self, class_list=list_of_classes):
        """
        :param class_list: list of instances of the ColumnOperations class
        :type  class_list: list
        """

        self.class_list = class_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # drop all columns that has DROP_ in its column name
        X = drop_col_with_DROP(X)
        # use the list of classes and perform operations on each column based on the classes
        X_transformed = perform_operations_with_classes(X, self.class_list)
        return X_transformed


#======================================================================
# Create pipeline for standard scalar and one hot encoding.
#======================================================================


def split_cat_num(df):
    """
    Split the dataframe into two dataframes, 
    where one contains the numeric datatypes 
    and the other contains categorical datatypes.

    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :returns: (df with numeric dtypes, df with categorical dtypes)
    :rtype:   (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
    """

    col_names = set(df.columns)
    types = set([str(dtype) for dtype in df.dtypes.values])

    num_cols = df.select_dtypes(
        include=['int8', 'int16', 'int32', 'int64', 'float64'])
    cat_cols = df.select_dtypes(include=['category'])

    num_col_names = set(num_cols.columns)
    cat_col_names = set(cat_cols.columns)

    missing_col_names = col_names.difference(
        num_col_names).difference(cat_col_names)

    if len(missing_col_names) != 0:
        print("Columns Missing:", missing_col_names)

    return num_cols, cat_cols

def transform_df(df, cat_encode_type):
    """
    :param df: dataframe
    :type  df: pandas.core.frame.DataFrame
    :param cat_encode_type: The type of encoding (one hot or just label)
    :type  cat_encode_type: str 
    :returns: (numpy array of size (instances, 
              number of numeric columns + category one hot encoded columns),
              pipeline)
    :rtype:   (numpy.ndarray,
               sklearn.compose._column_transformer.ColumnTransformer)
    """

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    num_cols, cat_cols = split_cat_num(df)
    num_col_names = list(num_cols.columns)
    cat_col_names = list(cat_cols.columns)

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    
    if cat_encode_type == "one hot":
        # one hot encoded: a lot more columns
        from sklearn.preprocessing import OneHotEncoder
        full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_col_names),
        ("cat", OneHotEncoder(), cat_col_names),
        ])
        result = full_pipeline.fit_transform(df).toarray()
        
    elif cat_encode_type == "numeric":
        from sklearn.preprocessing import OrdinalEncoder
        full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_col_names),
        ("cat", OrdinalEncoder(), cat_col_names),
        ])
        result = full_pipeline.fit_transform(df)

    else:
        raise ValueError("Not an available encoder type")

    return result, full_pipeline


#======================================================================
# Dealing with labels
#======================================================================


def encode_labels(labels, convert_type):
    """
    Convert labels into numeric labels (either numeric or one hot)

    :param labels: a series of the labels
    :type  labels: pandas.core.series.Series
    :param labels: type of conversion
    :type  labels: str
    :returns: (encoder, array of the encoded result)
    :rtype:   (sklearn.preprocessing.label.LabelEncoder,
               numpy.ndarray)
    """

    labels = pd.DataFrame(labels)
    if convert_type == "numeric":
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
    elif convert_type == "one hot":
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()
    else:
        raise ValueError("Not an avaliable convert type")

    label_encoded = encoder.fit_transform(labels)
    return encoder, label_encoded