"""
This file is for visualizing the results of trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import xgboost as xgb


#======================================================================
# XGB Models
#======================================================================

"""
XGB does not explicitly handle categorical data.
We either convert the categories to numbers or one hot encode it.

Cons of converting to numbers:
    introduces ordering where it shouldn't exist

Cons of one hot encoding:
    when there are many categories for a feature, 
    XGB is slow and the tree algorithm downplays the 1hot encoded features

Reminder: there are 3 outputs
"""


def xgb_feature_importance(xgc):
    """
    Plot feature importance from the xgb model.
    Measures include: wieght, split mean gain, sample converage
    
    :param xgc: the trained model of xgb
    :type  xgc: xgboost.sklearn.XGBClassifier
    """
    
    fig = plt.figure(figsize = (16, 12))
    title = fig.suptitle("Default Feature Importances from XGBoost", fontsize=14)

    ax1 = fig.add_subplot(2,2, 1)
    xgb.plot_importance(xgc, importance_type='weight', ax=ax1)
    t=ax1.set_title("Feature Importance - Feature Weight")

    ax2 = fig.add_subplot(2,2, 2)
    xgb.plot_importance(xgc, importance_type='gain', ax=ax2)
    t=ax2.set_title("Feature Importance - Split Mean Gain")

    ax3 = fig.add_subplot(2,2, 3)
    xgb.plot_importance(xgc, importance_type='cover', ax=ax3)
    t=ax3.set_title("Feature Importance - Sample Coverage")
    plt.show()


#======================================================================
# General Models
#======================================================================


def test_measure(answer, prediction, metric):
    """
    Use a metric to measure the success of the model
    
    :param answer: The label answer
    :type  answer: numpy.ndarray
    :param prediction: The predicted answer
    :type  prediction: numpy.ndarray
    :param answer: The metric used to evaluate the model
    :type  metric: str
    :returns: the evaluation metric number
    :rtype:   float
    """
    
    if metric == "accuracy":
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(answer, prediction)
        accuracy_perc = accuracy * 100.0
        print("Accuracy: %.2f%%" % accuracy_perc)
        return accuracy_perc
    else:
        raise ValueError("Not an avaliable metric")


def eli_instance(answer, prediction, label_value, match=False):
    
    """
    Get an index where the label is a certain value
    
    :param answer: the numpy array of the label answers
    :type  answer: numpy.ndarray 
    :param prediction: the numpy array of the label predictions
    :type  prediction: numpy.ndarray 
    :param label_value: the prediction value
    :type  label_value: int
    :param match: whether the prediction and answer matches or not
    :type  match: bool
    :returns: index of one of the label
    :rtype:   num
    """
    
    if match:
        in_pred_and_actual = np.logical_and (answer == label_value, prediction == label_value)
        index = np.where(in_pred_and_actual)[0]
    else:
        index = np.where(answer == label_value)[0]
    doc_num = random.choice(index)

    print('Actual Label:', answer[doc_num])
    print('Predicted Label:', prediction[doc_num])

    return doc_num


def inc_pca (X):
    """
    Perform incremental PCA to reduce dimensions while keeping high variance.
    """
    
    n_batches = 100
    inc_pca = IncrementalPCA (n_components = 14*14)
    for X_batch in np.array_split (X, n_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced = inc_pca.transform(X)
    return (X_reduced)


def conf_mx_rates (y, y_pred):
    """
    Given labels and predictions, creates a confusion matrix of error rates.
    Each row is an actual class, while each column is a predicted class.
    The whiter the square, the more the image is misclassified

    :param y: The labels
    :type  y: pandas.core.series.Series
    :param y_pred: The predictions based on the ML algorithm.
    :type  y_pred: pandas.core.series.Series
    """

    from sklearn.metrics import confusion_matrix
    
    conf_mx = confusion_matrix (y, y_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()


#======================================================================
# Validation Set Evaluations
#======================================================================


# scoring types
f1 = 'f1_macro' 
acc = 'accuracy'
cv=3


class Evaluations:
    """
    A class for storing the evaluation results of each model
    """

    def __init__(self, model_name, validation_score, y_pred):
        """
        :param model_name: name of the model
        :type  model_name: str
        :param validation_score: the result of evaluating the model on the validation set
        :type  validation_score: 
        :param y_pred: the predictions based on the model
        :type  y_pred: numpy.ndarray
        """

        self.model_name = model_name
        self.validation_score = validation_score
        self.y_pred = y_pred


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


def report_scores(model, valid_X, valid_y, encoder, model_name):
    """
    Report the scores of a certain model.
    Includes:
        validation score
        confusion matrix
        classification report

    :param model: sklearn model
    :type  model: sklearn model
    :param valid_X: inputs
    :type  valid_X: numpy.ndarray
    :param valid_y: labels
    :type  valid_y: numpy.ndarray
    :param encoder: sklearn encoder that codes categories with numeric
    :type  encoder: sklearn.preprocessing.label.LabelEncoder
    :param model_name: name of the model
    :type  model_name: str
    :returns: a class containing the evaluation attributes
    :rtype:   Evaluations class
    """

    # valid_score(model, valid_X, valid_y)
    
    y_pred = model.predict(valid_X)
    acc = test_measure(valid_y, y_pred, metric="accuracy")

    target_names = list(encoder.classes_)
    print("Classes: ", target_names)

    print("Confusion Matrix: ")
    conf_mx_rates(valid_y, y_pred)

    print("Classification Report: ")
    from sklearn.metrics import classification_report
    print(classification_report(
        valid_y,
        y_pred,
        target_names=target_names
    ))

    evaluation_class = Evaluations(model_name, acc, y_pred)

    return evaluation_class
