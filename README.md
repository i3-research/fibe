# Efficient Feature Selection Using Forward Inclusion & Backward Elimination (FIBE)

This algorithm performs nested cross-validation feature selection for both regression and classification tasks using Forward Inclusion Backward Elimination (FIBE). The algorithm uses nested cross-validation where N outer folds divide the data, and for each outer fold, N inner folds perform cross-validation to select features. Multiple voting strategies are available to aggregate features across outer folds.

## GitHub Repo
https://github.com/i3-research/fibe

## Features

### Supported Models
- **Regression**: linearSVR, gaussianSVR, RegressionForest, AdaBoostDT, AdaBoostSVR, and consensus (using linearSVR, gaussianSVR, and RegressionForest)
- **Classification**: linearSVC, gaussianSVC, RandomForest, AdaBoostDT, AdaBoostSVC, and consensus (using linearSVC, gaussianSVC, and RandomForest)

### Performance Metrics
- **Regression**: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE)
- **Classification**: Accuracy, F1-score, binaryROC

### Voting Strategies
The algorithm provides seven voting strategies to select the final feature set from N outer fold selections:

1. **'strict'**: Features selected at least 0.6 x N times across outer folds
2. **'loose'**: Features selected at least 0.4 x N times across outer folds
3. **'weighted'** (default): Weighted ranking based on feature positions in each fold's selected feature list, with threshold at max_length (Km), and ensures features selected ≥3 times (strict) are included
4. **'union'**: Takes the union of all features selected across all N outer folds
5. **'conditional'**: First tries strict voting, then falls back to loose voting, and finally to union based on specific conditions
6. **'2-stage-selection-with-union'**: First stage takes union of features from N outer folds, then reruns the entire FIBE process on these features with reshuffled data partitions (different random seed) to produce a second set of N feature selections, and finally takes union of the second stage features as the final selection
7. **'2-stage-selection-with-weighted-voting'**: First stage takes union of features from N outer folds, reruns the FIBE process on these features with reshuffled data partitions, and then applies weighted majority voting across all feature sets from both stages (total 2 x N lists) to determine the final feature subset
8. **'best-fold'**: Evaluates each outer fold's selected features on all other (N-1) outer folds using N inner folds cross-validation on each, computes mean performance (accuracy/error) for each fold, and selects the fold with best mean performance (highest for classification, lowest for regression) as the final feature set

## How to Run the Algorithm

To run this algorithm, call the `fibe()` function with appropriate parameters:

```python
final_features, subjectList, actualScore, predictedScore, validationPerformance, dfw = fibe(
    feature_df, 
    score_df, 
    data_cleaning=False, 
    fixed_features=None, 
    columns_names=None, 
    task_type=None, 
    probability=False, 
    balance=False, 
    model_name=None, 
    metric=None, 
    voting_strictness=None, 
    nFold=None, 
    maxIter=None, 
    tolerance=None, 
    maxFeatures=None, 
    save_intermediate=False, 
    output_dir=None, 
    inference_data_df=None, 
    inference_score_df=None, 
    verbose=True
)
```

## Parameters

### Required Parameters
- **`feature_df`**: The 2D feature matrix (supports DataFrame, Numpy Array, and List) with columns representing different features.
- **`score_df`**: The 1D score vector as a column (supports DataFrame, Numpy Array, and List).

### Optional Parameters

#### Data Processing
- **`data_cleaning`** (bool, default=False): If True, cleans the data including dropping invalid and imbalanced features, mapping categories to numeric values, imputing data with median/mean values.
- **`fixed_features`** (list or DataFrame, default=None): Predefined features that must stay in the feature set and the FIBE algorithm does not add or remove those. Must be either a List of names to select from `feature_df`, or DataFrame of features added separately to `feature_df`.
- **`columns_names`** (list, default=None): Contains the names of the features. The algorithm returns the names of the selected features from this list. If not available, then the algorithm returns the column indexes of selected features.

#### Task Configuration
- **`task_type`** (str, default='regression'): Either `'regression'` or `'classification'`.
- **`probability`** (bool, default=False): If True, probability values (for class 1) that lead to binary classifications is returned. This option only works when `task_type` is `'classification'`.
- **`balance`** (bool, default=False): In a binary classification task, if the data is imbalanced in terms of classes, `balance=True` uses resampling to balance the data.

#### Model Selection
- **`model_name`** (str, default=None): 
  - For `'regression'` task: `'linearSVR'`, `'gaussianSVR'`, `'RegressionForest'`, `'AdaBoostDT'`, `'AdaBoostSVR'`, or `'consensus'`. Default is `'linearSVR'`.
  - For `'classification'` task: `'linearSVC'`, `'gaussianSVC'`, `'RandomForest'`, `'AdaBoostDT'`, `'AdaBoostSVC'`, or `'consensus'`. Default is `'linearSVC'`.

#### Performance Metrics
- **`metric`** (str, default=None):
  - For `'regression'` task: `'MAE'` or `'MAPE'`. Default is `'MAE'`.
  - For `'classification'` task: `'Accuracy'`, `'F1-score'`, or `'binaryROC'`. Default is `'Accuracy'`.

#### Feature Selection Configuration
- **`voting_strictness`** (str, default='weighted'): Choose from `'strict'`, `'loose'`, `'weighted'`, `'union'`, `'conditional'`, `'2-stage-selection-with-union'`, `'2-stage-selection-with-weighted-voting'`, or `'best-fold'`. See [Voting Strategies](#voting-strategies) section for details.
- **`nFold`** (int, default=5): Number of folds in cross-validation. Preferred and default is `5`.
- **`maxIter`** (int, default=3): Maximum number of iterations that the algorithm goes back and forth in forward inclusion and backward elimination in each fold.
- **`tolerance`** (float, default=0.05): Percentage of deviation in the error/accuracy threshold allowed. Default is `0.05` (5%).
- **`maxFeatures`** (int, default=3): Number of features to be allowed under tolerance. Default is `3`.

#### Output Options
- **`save_intermediate`** (bool, default=False): If True, saves intermediate results to the specified directory.
- **`output_dir`** (str, default=None): Directory where intermediate results are saved if `save_intermediate` is True.
- **`inference_data_df`** (DataFrame/Array/List, default=None): Data for optional second inference cohort for prediction using the selected subset of features.
- **`inference_score_df`** (DataFrame/Array/List, default=None): Scores for optional second inference cohort for prediction using the selected subset of features.
- **`verbose`** (bool, default=True): Generates text for intermediate loss and selected feature list during iteration.

## Outputs

The function returns a tuple with the following elements (in order):

1. **`final_features`** (list): The list of features if `columns_names` was not `None`. Otherwise column indexes of the selected features. This represents the final selected feature set after applying the chosen voting_strictness method.

2. **`subjectList`** (list): List of subjects used in inference. Each subject/patient is assigned a name as `'subXX'` and according to this list, other outputs are organized in the subsequent generated lists.

3. **`actualScore`** (list): List containing actual target scores. If `model_name` is chosen as `'consensus'`, this list has a repetition of values 3 times, to correspond to predictions by three models. If `save_intermediate` is `True`, `actualScore[-1]` contains an additional list of actual score values of the inference data (if `inference_data_df` is provided).

4. **`predictedScore`** (list or nested list): List containing predicted scores. If `model_name` is chosen as `'consensus'`, this list has 3 predictions per observation. Although 3 predictions per observation are generated here, `'consensus'` uses an averaging of the losses for 3 predictions in decision-making. If `probability` is set `True` and `task_type` is `'classification'`, then `predictedScore` contains an additional list of prediction probability for class 1 score values. The structure is then `[predicted, predicted_probs]`.

5. **`validationPerformance`** (list): List containing validation performance in terms of chosen `metric` for `nFold` folds. Each element corresponds to the performance on one fold during cross-validation inference.

6. **`dfw`** (DataFrame or None): DataFrame containing feature weights (for `'weighted'` voting_strictness) or None (for other voting methods). When available, it contains columns: `'Feature'`, `'Weight'`, and `'Relative Weight (%)'` sorted by relative weight in descending order.

## Algorithm Overview

![FIBE Algorithm Schematic](figure/figure_v2.png?raw=true "FIBE Algorithm")

Figure: Schematic diagram of the FIBE algorithm showing nested cross-validation structure with N outer folds and N inner folds for feature selection.

## Example Code

An example Python file `main.py` is included. It demonstrates:
- Running classification problems
- Running regression problems
- Using predefined fixed features
- Using data balancing options
- Different voting strategies

## Required Packages

Please refer to the `requirements.txt` file for all required dependencies.

## Citation

If you use this software, please cite:

```
Forward Inclusion Backward Elimination (FIBE) Algorithm
Copyright (c) Yangming Ou, Boston Children's Hospital/Harvard Medical School, 2023
```

## Contact and Support

For suggestions, comments, or questions, please contact:
- **Prof. Yangming Ou, PhD**: Yangming.Ou@childrens.harvard.edu
- **Mohammad Arafat Hussain, PhD**: Mohammad.Hussain@childrens.harvard.edu

## License

See the copyright notice in the source code files for licensing information.
