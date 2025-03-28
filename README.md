# Efficient Feature Selection Using Forward Inclusion & Backward Elimination (FIBE)
This algorithm performs a feature selection for both regression and classification tasks using any of the following models (1) linear support vector regressor/classifier, (2) Gaussian support vector regressor/classifier, (3) Regression/random forest, (4) AdaBoost regressor/classifier with linear support vector and (5) AdaBoost regressor/classifier with decision trees. This algorithm can also use model consensus in feature selection using (1) linear support vector regressor/classifier, (2) Gaussian support vector regressor/classifier, and (3) Regression/random forest. For loss calculation as well as validation performance estimation, this algorithm comes with options of using (1) mean absolute error (MAE) (2) mean absolute percentage error (MAPE), (3) accuracy, (4) F1-score, and (5) binaryROC metrics. The ``fibe_function`` contains all the necessary functions to run this algorithm. 

## GitHub Repo
https://github.com/i3-research/fibe

## Install

You can install this package with:

```
pip install -e git+https://github.com/i3-research/fibe.git#egg=fibe
```

or add the following into your `pyproject.toml` dependencies:

```
fibe @ git+https://github.com/i3-research/fibe.git
```

## How to Run the Algorithm
To run this algorithm, the following function is needed to call with appropriate parameter selection:

```python
from fibe import fibe

selectedFeatures, actualScore, predictedScore, validationPerformance = fibe(feature_df, score_df, data_cleaning=False, fixed_features=None, columns_names=None, task_type=None, probability=False, balance=False, model_name=None, metric=None, voting_strictness=None, nFold=None, maxIter=None, tolerance=None, maxFeatures=None, save_intermediate=False, output_dir=None, inference_data_df=None, inference_score_df=None, verbose=True, model_kwargs=None)
```

Here, 
- ``feature_df`` is the 2D feature matrix (supports DataFrame, Numpy Array, and List) with columns representing different features.
- ``score_df`` is the 1D score vector as a column (supports DataFrame, Numpy Array, and List).
- ``data_cleaning`` if True, cleans the data including dropping invalid and imbalanced features, mapping categories to numeric values, imputing data with median/mean values.
- ``fixed_features`` Predefined features that must stay in the feature set and the FIBE algorithm does not add or remove those. Must be either a List of names to select from 'feature_df,' or DataFrame of features added separately to 'feature_df.'
- ``columns_names`` contain the names of the features. The algorithm returns the names of the selected features from this list. If not available, then the algorithm returns the column indexes of selected features. 
- ``task_type`` either 'regression' or 'classification.' The default is 'regression.'
- ``probability`` if True, probability values (for the class 1) that leads to binary classifications is returned. This option only works when the 'task_type=regression.'
- ``balance`` In a binary classification task, if the data is imbalanced in terms of classes, 'balance=True' uses resampling to balance the data.
- ``model_name`` For 'regression' task, choose from 'linearSVR', 'gaussianSVR', 'RegressionForest', 'AdaBoostDT', 'AdaBoostSVR', and 'consensus' (consensus using 'linearSVR', 'gaussianSVR', and 'RegressionForest'). The default is ``'linearSVR'``. For the 'classification' task, choose from 'linearSVC', 'gaussianSVC', 'RandomForest', 'AdaBoostDT', 'AdaBoostSVC', and 'consensus' (consensus using 'linearSVC', 'gaussianSVC', and 'RandomForest'). The default is ``'linearSVC'``.
- ``metric`` For the ``regression`` task, choose from 'MAE' and 'MAPE'. The default is 'MAE.' For the 'classification' task, choose from 'Accuracy', 'F1-score', and 'binaryROC'. The default is ``'Accuracy'``.
- ``voting_strictness`` The option 'strict' chooses those features that are selected at least 3 times in 5-fold cross-validation; the option 'loose' chooses those features that are selected at least 2 times in 5-fold cross-validation, and the option 'both' produces two sets of results, one for 'strict' and one for 'loose'. The default is ``'strict'``. For any random number of folds, ``N``, the 'strict' threshold should be ``0.6 X N`` and the 'loose' threshold should be ``0.4 X N``.
- ``nFold`` Number of folds in cross-validation. Preferred and default is ``5``.
- ``maxIter`` is the maximum number of iterations that the algorithm goes back and forth in forward inclusion and backward elimination in each fold. The default is ``3``.
- ``tolerance`` is the percentage of deviation in the error/accuracy threshold allowed. The default is ``0.05``, i.e., 5%.
- ``maxFeatures:`` is the fractional number that indicate the number of best features to be selected of the total features. Default is ``0.25``, i.e., 25% of the total number of features.
- ``save_intermediate`` If True, saves intermediate results to the specified directory. Default is False.
- ``output_dir`` Directory where intermediate results are saved if save_intermediate is True.
- ``inference_data_df`` Data for optional second inference cohort for prediction using the selected subset of features.
- ``inference_score_df`` Scores for optional second inference cohort for prediction using the selected subset of features.
- ``verbose`` generates text for intermediate loss and selected feature list during iteration. The default is ``True``.
- ``model_kwargs`` If not ``None``, keyword-based argments for the model. Please refer to code to see the settings for consense. The default is ``None``.

The outputs are:
- ``selectedFeatures`` is the list of features if ``columns_names`` was not ``None``. Otherwise column indexes of the selected features. For ``voting_strictness`` of 'both', ``selectedFeatures`` contains two sets of output as ``[[selected features for 'strict'], [selected feature for 'loose']]``. 
- ``actualScore`` is the list containing actual target scores. If ``model_name`` is chosen as 'consensus', this list has a repetition of values 3 times, to correspond to predictions by three models. For ``voting_strictness`` of 'both', ``actualScore`` contains two sets of output as ``[[actual scores for 'strict'], [actual scores for 'loose']]``. 
- ``predictedScore`` is the list containing predicted scores. If ``model_name`` is chosen as 'consensus', this list has 3 predictions per observation. Although 3 predictions per observation are generated here, 'consensus' uses an averaging of the losses for 3 predictions in decision-making. For ``voting_strictness`` of 'both', ``predictedScore`` contains two sets of output as ``[[predicted scores for 'strict'], [predicted score for 'loose']]``. If the argument ``probability`` is set 'True' and ``task_type`` is 'cassification', then ``predictedScore`` contains an additional list of prediction probability for class 1 score values for the inference data. The structure is then ``[[[predicted scores for 'strict'], ['predicted probabilities for 'strict']], [[predicted score for 'loose'],[predicted probabilities for 'loose']]``.
- ``validationPerformance`` is a list containing validation performance in terms of chosen ``metric`` for ``nFold`` folds. For ``voting_strictness`` of 'both', ``validationPerformance`` contains two sets of output as ``[[validation performance for 'strict'], [validation performance score for 'loose']]``. 

## Algorithm Overview
![Alt text](figure/figure_v2.png?raw=true "Title")
Figure: Schematic diagram of our FIBE algorithm.

## Example Code
An example Python file ``main.py`` is given. It includes example code to run one classification and one regression problem. Further, it includes examples of how to run the algorithm with predefined fixed features as well as data balancing options.

## Required Packages
Please refer to the ``pyproject.toml`` file.

## Suggestions and Comments
- Prof. Yangming Ou, PhD (Yangming.Ou@childrens.harvard.edu)
- Mohammad Arafat Hussain, PhD (Mohammad.Hussain@childrens.harvard.edu)
