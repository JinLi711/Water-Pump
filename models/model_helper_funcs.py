"""
This file is for helper functions that are used in the models.


======================================================================
Brief Description of Each Model
======================================================================


XGB (Extreme Gradient Boosting)
    Create decision tree.
    Look at residual errors made by first predictor.
    Train another decision tree on the errors.
    Repeat.
    For predictions, add up predictions for all trees.

SGD (Stochastic Gradient Descent)
    Start with randomized weights.
    Take an instance one at a time.
    "Run" it through the weights.
    Compute loss.
    Update weights by going down the gradient.

Logistic (Binary)
    Eastimate the probability that an instance is in a class
    with the logistic function (curvy one with min=0, max=1).
    Compute loss, and go down gradient descent.

Softmax Regression
    Not really a regression.
    Compute score for each possible class.
    Estimate probability for each class.
    Pick class with highest score.
    Gradient descent down loss function.

Linear SVM (Support Vector Machine) Classifier
    draw a line that seperates classes as much as possible
    Make the street as thick as possible with as little infringement

Decision Trees
    Start at root node.
    Split tree based on certain condition.
    Run down the tree to see prediction.

Random Forests
    Split training set
    Train each set on a different decision tree.
    Have each predict.
    The final value is the most frequent prediction.

Extremely Randomized Trees Ensemble
    use random thresholds rather than search for best possible thresholds.

AdaBoost
    Start with base classifier.
    Make prediction.
    Relative weight of missclassified instances increase.
    Repeat with second classifier using updated weights.
    Repeat.


======================================================================
Brief Description of Metrics
======================================================================


Precision
    ability of the classifier not to label as positive a sample that is negative

Recall
    ability of the classifier to find all the positive samples

F1-Score
    weighted harmonic mean of precision and recall

Support
    number of samples of the true response that lie in that class

Micro Average
    average the total true positives, false negatives and false positives

Macro Average
    average the unweighted mean per label

Weighted Average
    average the support-weighted mean per label
"""


#======================================================================
# Random Search For Best Hyperparameters
#======================================================================


# scoring types
f1 = 'f1_macro' 
acc = 'accuracy'
cv=3

def rand_search (model, param_grid, X, y, scoring=acc, cv=cv):
    """
    Use randomized search to find the best hyperparameters
    
    :param model: a model that can be entered in random search 
    :type  model: sklearn model
    :param param_grid: dictionary mapping hyperparameter names to a 
                       list of values to try
    :type  param_grid: dict
    :param X: the input
    :type  X: numpy.ndarray
    :param y: label
    :type  y: numpy.ndarray
    :param scoring: the scoring methodology
    :type  scoring: str
    :param cv: number of sets to split for cross validation
    :type  cv: int
    """
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    
    grid_search = RandomizedSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring=scoring
    )
    grid_search.fit(X, y )
    
    c_val_scores = cross_val_score(
        grid_search, 
        X, 
        y, 
        cv=cv, 
        scoring=scoring
    )
    
    print("Best parameters:\n", grid_search.best_params_)
    print("Cross Validation scores:\n", c_val_scores)
    
    return grid_search


#======================================================================
# Validation set score
#======================================================================


def valid_score(model, X, y, scoring=acc, cv=cv):
    """
    Find the validation score.
    
    :param model: a model that can be entered in random search 
    :type  model: sklearn model
    :param X: the validation input
    :type  X: numpy.ndarray
    :param y: validation label
    :type  y: numpy.ndarray
    :param scoring: the scoring methodology
    :type  scoring: str
    :param cv: number of sets to split for cross validation
    :type  cv: int
    """
    
    from sklearn.model_selection import cross_val_score
    
    score = cross_val_score(
        model, 
        X, 
        y, 
        cv=cv, 
        scoring=scoring
    )
    print("Validation Set Scores:\n", score)