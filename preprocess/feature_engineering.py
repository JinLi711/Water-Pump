"""
Some ideas for feature engineering:

Numeric:
Binarization: turn data into binary data.
Instead of linear weights, we can have polynomial weights.
Binning: put numeric values into bins, especially useful for inputs that have outliers or large ranges
    fixed binning: decide on the bin sizes myself
    adaptive binning (usually safer): can divide bins into quartiles 
Numeric Transformations: 
    log: best used for skewed distributions as it tends to expand the values that fall in the range of lower magnitudes and compress the range of higher magnitudes
    box cox: used when variabity in different regions are largely different
    arcsin: for porportions
Outlier Removals:
    z-score: parametric, usually 2.5, 3, 3.5
    Dbscan:  https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561
    Isolation Forest:
Dealing With Irregular Values:
    replace by mean or median: problem is that it doesn't account for uncertainty in imputations and doesn't factor in correlations between features
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
    Note that this is very tricky, and I really need to compare before and afters for training to see if this works.
"""