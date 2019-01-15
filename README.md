# Water-Pump

Repository For Predicting the Conditions of Water Pumps (either functional, not functional, or functional but needs repairs). To see more about the data, check [here](https://github.com/JinLi711/Water-Pump/blob/master/Context.txt).

Still in working progress.

# Visualization

Visualize individual features independently using histograms. Did not plot a histogram for categories with too many features.

Visualized correlations between categorical features. I used cramers V and theils U make heat maps.

Visualized correlations between numeric features and cateogirical features using correlation ratio. Used heat maps.

Visualized correlations between numeric features with pearson's R. Created heat maps, joint plots, and pairplots.

Visualization is mainly used to perform feature extraction and understanding the data.

# Preprocess

  * Dealing with all datatypes:
      * dropping irrelevant features
  * Dealing with numerics:
      * change zero values by:
          * dropping rows with zeros in it
          * replacing zeros with median
      * scaling numeric values like:
          * log scale
          * box cox scale
      * absolute value negative values
      * binning values
      * removing outliers with Z-score threshold
      * standard scale numeric values

  * Dealing with datetime:
      * convert datetime to days from first date

  * Dealing with categories:
      * replace NaN with the most common category
      * reduce dimensions of categorical data
      * convert categories into one-hot encoding

# Models Used

  * XGB
  * SGD
  * Softmax Logistic Regression
  * Linear SVM
  * Decision Trees
  * Random Forests
  * AdaBoost
  * Ensemble