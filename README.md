# Water-Pump

Repository For predicting the conditions of water pumps in Tanzania (either functional, not functional, or functional but needs repairs). To see more about the data, check [here](https://github.com/JinLi711/Water-Pump/blob/master/Context.txt).


# Visualization

Visualized:

  * individual features independently 
  * correlations between categorical features (using [cramers V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) and [theils U](https://en.wikipedia.org/wiki/Uncertainty_coefficient))
  * correlations between numeric features and cateogirical features (using [correlation ratio](https://en.wikipedia.org/wiki/Correlation_ratio))
  * correlations between numeric features (using [pearson's R](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient))
  
Created histograms, heatmaps, joint plots, and pairplots.

Visualization is mainly used to perform feature extraction and understanding the data.

# Preprocess

  * 80/20 split for train and validation. For the final model, the model was retrained on the entire train set.

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

# Results

80% accuracy. For reference, the best results produced about 82% accuracy.

# Problems

  * Dropping rows could not be used because I could not turn this off when applying the transformation to the final test set (I'm not allowed to drop rows for the final test set).
