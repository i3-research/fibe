# Efficient Feature Selection Using Forward Inclusion & Backward Elimination (FIBE)
This algorithm performs a feature selection for both regression and classification tasks using any of following models (1) linear support vector regressor/classifier, (2) Gaussian support vector regressor/classifier, (3) Regression/random forest, (4) AdaBoost regressor/classifier with linear support vector and (5) AdaBoost regressor/classifier with decision trees. This algorithm can also use model consensus in feature selection using (1) linear support vector regressor/classifier, (2) Gaussian support vector regressor/classifier, and (3) Regression/random forest. For loss calculation as well as validation performance estimation, this algorithm comes with options of using (1) mean absolute error (MAE) (2) mean absolute percentage error (MAPE), (3) accuracy, and (4) binaryROC metrics. The ``fibe_function`` contains all the necessary functions to run this algorithm. 

## How to Run the Algorithm
To run this algorithm, the following function is needed to call with appropriate parameter selection:

``selectedFeatures, validationPerformance = fibe(feature_df, score_df, columns_names=None, task_type=None, model_name=None, metric=None, voting_strictness=None, nFold=None, maxIter=None, verbose=True)``

Here, 
- ``feature_df`` is the 2D feature matrix (supports DataFrame, Numpy Array, and List) with columns representing different features.
- ``score_df`` is the 1D score vector as a column (supports DataFrame, Numpy Array, and List).
- ``columns_names`` contain the names of the features. The algorithm returns the names of the selected features from this list. If not available, then the algorithm returns the column indexes of selected features. 
- ``task_type`` either 'regression' or 'classification.' Default is 'regression.'
- ``model_name`` For ``regression`` task, to choose from 'linerSVR', 'gaussianSVR', 'RegressionForest', 'AdaBoostDT', 'AdaBoostSVR', and 'consensus' (consensus using 'linerSVR', 'gaussianSVR', and 'RegressionForest'). Default is ``'linerSVR'``. For ``classification`` task, to choose from 'linerSVC', 'gaussianSVC', 'RandomForest', 'AdaBoostDT', 'AdaBoostSVC', and 'consensus' (consensus using 'linerSVC', 'gaussianSVC', and 'RandomForest'). Default is ``'linerSVC'``.
- ``metric`` For ``regression`` task, to choose from 'MAE' and 'MAPE'. Default is ``'MAE'``. For ``classification`` task, to choose from 'Accuracy', and 'binaryROC'. Default is ``'Accuracy'``.
- ``voting_strictness`` either 'strict' that chooses those features that is selected at least 3 times in 5-fold cross-validation, or 'loose' that chooses those features that is selected at least 2 times in 5-fold cross-validation. Default is ``'strict'``.
- ``nFold`` Number of folds in cross-validation. Preferred and default is ``5``.
- ``maxIter`` is the maximum number of iteration that the algorithm goes back and forth in forward inclusion and backward elimination in each fold. Default is ``3``.
- ``verbose`` generates text for intermediate loss and selected feature list during iteration. Default is ``True``.

The outputs are:
- ``selectedFeatures`` is the list of features if ``columns_names`` was not ``None``. Otherwise column indexes of the selected features.
- ``validationPerformance`` is a list containing validation performance in terms of chosen ``metric`` for ``nFold`` folds.

An example caller python file ``main.py`` is given.

## Detailed Explanation of the Algorithm
TBD
