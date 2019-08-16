# Water Pump Functionality Prediction

This is a repository for predicting the conditions of water pumps in Tanzania (either functional, not functional, or functional but needs repairs) given categorical and numerical features.




## Motivation

One in every six people lack access to safe drinking water. The ability to predict which water pump will fail can improve maintenance operations and ensure people have access to clean water. 






## Visualization

Visualization is used to perform feature extraction and understanding the data.

Visualized:

* individual features independently 
* correlations between categorical features (using [cramers V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) and [theils U](https://en.wikipedia.org/wiki/Uncertainty_coefficient))
* correlations between numeric features and cateogirical features (using [correlation ratio](https://en.wikipedia.org/wiki/Correlation_ratio))
* correlations between numeric features (using [pearson's R](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient))
  
Created histograms, heatmaps, joint plots, and pairplots.



Click [here](https://github.com/JinLi711/Water-Pump/blob/master/data.md) to see more about the data. 

Click [here](https://github.com/JinLi711/Water-Pump/tree/master/visualization) for notebooks that visualizes the data.











## Method


### Preprocess

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

Click [here](https://github.com/JinLi711/Water-Pump/blob/master/input_actions.md) for more specifics.



### Models Used

  * XGB
  * SGD
  * Softmax Logistic Regression
  * Linear SVM
  * Decision Trees
  * Random Forests
  * AdaBoost








## Results

80% accuracy. For reference, the best results produced about 82% accuracy.

## Problems

  * Dropping rows could not be used because I could not turn this off when applying the transformation to the final test set (I'm not allowed to drop rows for the final test set).


## Acknowledgements

Data is gathered from Taarifa, an organization that gathered data from the Tanzania Ministry of Water.
The classification idea is from drivendata.org.