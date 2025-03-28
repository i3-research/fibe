# -*- coding: utf-8 -*-

# Forward Inclusion Backward Elimination
# Copyright (c) Mohammad Arafat Hussain, Boston Children's Hospital/Harvard Medical School, 2023

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
#
# Last Updated: 09/26/2024 at 2215H EST, By Mohammmad Arafat Hussain.


from typing import Optional
from .types import Metric, TaskType, ModelName, StrictType

import math
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

from . import data_curation
from . import train
from . import inference
from . import inference_additional


def fibe(
        feature_df: pd.DataFrame,
        score_df: pd.DataFrame,
        data_cleaning=False,
        fixed_features: Optional[list[str]] = None,
        columns_names: Optional[list[str]] = None,
        task_type: Optional[TaskType] = 'regression',
        probability=False,
        balance=False,
        model_name: Optional[ModelName] = None,
        metric: Optional[Metric] = None,
        voting_strictness: Optional[StrictType] = 'weighted',
        n_fold: Optional[int] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = 0.05,
        max_features: Optional[int] = None,
        save_intermediate=False,
        output_dir: Optional[str] = None,
        inference_data_df: Optional[pd.DataFrame] = None,
        inference_score_df: Optional[pd.DataFrame] = None,
        verbose=True,

        model_kwargs: Optional[dict] = None,
):
    '''
    feature_df: is the 2D feature matrix (supports DataFrame, Numpy Array, and List) with columns representing different features.
    score_df: is the 1D score vector as a column (supports DataFrame, Numpy Array, and List).
    data_cleaning: if True, cleans the data including dropping invalid and imbalanced features, mapping categories to numeric values, imputing data with median/mean values.
    fixed_features: Predefined features that must stay in the feature set and the FIBE algorithm does not add or remove those.
            Must be either a List of names to select from 'feature_df', or DataFrame of features added separately to 'feature_df.'
    columns_names: contain the names of the features. The algorithm returns the names of the selected features from this list.
            If not available, then the algorithm returns the column indexes of selected features.
    task_type: either 'regression' or 'classification.' The default is 'regression.'
    probability: if True, probability values (for the class 1) that leads to binary classifications is returned. This option only works when the 'task_type=regression.'
    balance: In a binary classification task, if the data is imbalanced in terms of classes, 'balance=True' uses resampling to balance the data.
    model_name: For the 'regression' task, choose from 'linearSVR', 'gaussianSVR', 'RegressionForest', 'AdaBoostDT', 'AdaBoostSVR',
            and 'consensus' (consensus using 'linearSVR', 'gaussianSVR', and 'RegressionForest'). The default is 'linearSVR'. For 'classification' task,
            to choose from 'linearSVC', 'gaussianSVC', 'RandomForest', 'AdaBoostDT', 'AdaBoostSVC', and 'consensus' (consensus using 'linerSVC',
            'gaussianSVC', and 'RandomForest'). The default is 'linearSVC'.
    metric: For the 'regression' task, choose from 'MAE' and 'MAPE'. The default is 'MAE.' For the 'classification' task, choose from 'Accuracy',
            'F1-score,' and 'binaryROC'. The default is 'Accuracy'.
    voting_strictness: either 'strict' that chooses those features that is selected at least 3 times in 5-fold cross-validation, or
            'loose' that chooses those features that is selected at least 2 times in 5-fold cross-validation, or 'both' that chooses both strict and loose options with some conditions. The default is 'strict'.
            For any random number of folds, N, the 'strict' threshold should be 0.6 X N and the 'loose' threshold should be 0.4 X N. When 'both' is selected, first the system checks if
            strick feature number is >= 2/3 of avarage number of features selected in 5-folds. If not, then check if loose feature number is <= 2, while the mean feature number over 5-folds
            is greater than 4, then the system does union of features.
    nFold: Number of folds in cross-validation. Preferred and default is '5'.
    maxIter: is the maximum number of iterations that the algorithm goes back and forth in forward inclusion and backward elimination in each fold. The default is '3'.
    tolerance: is the percentage of deviation in the error/accuracy threshold allowed. The default is '0.05', i.e., 5%.
    maxFeatures: is the fractional number that indicate the number of best features to be selected of the total features. Default is 0.25, i.e., 25% of the total number of features.
    save_intermediate: if True, saves intermediate results to the specified directory. Default is False.
    output_dir: directory where intermediate results are saved if save_intermediate is True.
    inference_data_df: data for optional second inference cohort for prediction using the selected subset of features.
    inference_score_df: scores for optional second inference cohort for prediction using the selected subset of features.
    verbose: generates text for intermediate loss and selected feature list during iteration. The default is 'True'.

    model_kwargs: keyword-based argments for the model. Please refer to code to see the settings for consense.

    The outputs are:
    selectedFeatures: is the list of features if 'columns_names' was not 'None'. Otherwise column indexes of the selected features. For 'voting_strictness' of 'both',
            'selectedFeatures' contains two sets of output as [[selected features for 'strict'], [selected feature for 'loose']].
    subjectList: is the list for subjects used in inference. Each subject/patient is assigned a name as 'subXX' and according to this list, other outputs are organized in the subsequent generated lists.
    actualScore: is the list containing actual target scores. If 'model_name' is chosen as 'consensus', this list has a repetition of values 3 times, to correspond to
            predictions by three models. For 'voting_strictness' of 'both', 'actualScore' contains two sets of output as [[actual scores for 'strict'], [actual scores for 'loose']]. If the argument
            'save_intermediate' is set 'True', 'actualScore[-1]' contains an additional list of actual score values of the inference data.
    predictedScore: is the list containing predicted scores. If 'model_name' is chosen as 'consensus', this list has 3 predictions per observation. Although 3 predictions per observation
            are generated here, 'consensus' uses an averaging of the losses for 3 predictions in decision-making. For 'voting_strictness' of 'both', 'predictedScore' contains two sets of
            output as [[predicted scores for 'strict'], [predicted score for 'loose']]. If the argument 'probability' is set 'True' and 'task_type' is 'cassification', then predictedScore contains
            an additional list of prediction probability for class 1 score values for the inference data. The structure is then
            [[[predicted scores for 'strict'], ['predicted probabilities for 'strict']], [[predicted score for 'loose'],[predicted probabilities for 'loose']]
    validationPerformance: is a list containing validation performance in terms of chosen 'metric' for 'nFold' folds. For 'voting_strictness' of 'both', 'validationPerformance' contains
            two sets of output as [[validation performance for 'strict'], [validation performance score for 'loose']].
    '''

    start_time = datetime.now()
    print("Code started running at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Data Curation : Added by Ankush Kesri
    if data_cleaning == True:
        feature_df, drop_log, mapping_list, imputation_log = data_curation.data_curation(feature_df)

        # Saving log files
        data_curation.log_files_generator(drop_log, mapping_list, imputation_log, output_dir)

    # Checking and converting to DataFrame if the features are in numpy array format
    if type(feature_df) is np.ndarray:
        if columns_names != None:
            # checking if the length of feature names is equal to the feature column numbers
            if feature_df.shape[1] != len(columns_names):
                raise ValueError(
                    "Number of columns in the Feature is not equal to the number of column names provided")
            else:
                feature_df = pd.DataFrame(data=feature_df, columns=columns_names)
        else:
            feature_df = pd.DataFrame(data=feature_df)

    # Checking and converting to DataFrame if the features are in list format
    elif isinstance(feature_df, list):
        if columns_names != None:
            # checking if the length of feature names are equal to the feature column numbers
            if len(feature_df[0]) != len(columns_names):
                raise ValueError(
                    "Number of columns in the Feature is not equal to the number of column names provided")
            else:
                feature_df = pd.DataFrame(data=feature_df, columns=columns_names)
        else:
            feature_df = pd.DataFrame(data=feature_df)

    # checking and converting to DataFrame if the target is in numpy array format
    if type(score_df) is np.ndarray:
        score_df = pd.DataFrame(data=score_df)

    # Checking and converting to DataFrame if the target is in list format
    elif isinstance(score_df, list):
        score_df = pd.DataFrame(data=score_df)

    # checking if the observation/subject numbers are equal in features and in targets
    if feature_df.shape[0] != score_df.shape[0]:
        raise ValueError(
            "Number of rows in the Feature is not equal to the number of rows in Score")

    if inference_data_df is not None and inference_score_df is None:
        raise ValueError("Scores for inference cohort is not provided.")

    if inference_data_df is not None and inference_score_df is not None:
        # Checking and converting to DataFrame if the inference features are in numpy array format
        if type(inference_data_df) is np.ndarray:
            if columns_names != None:
                # checking if the length of feature names is equal to the feature column numbers
                if inference_data_df.shape[1] != len(columns_names):
                    raise ValueError(
                        "Number of columns in the inference feature is not equal to the number of column names provided")
                else:
                    inference_data_df = pd.DataFrame(data=inference_data_df, columns=columns_names)
            else:
                inference_data_df = pd.DataFrame(data=inference_data_df)

        # Checking and converting to DataFrame if the features are in list format
        elif isinstance(inference_data_df, list):
            if columns_names != None:
                # checking if the length of feature names are equal to the feature column numbers
                if len(inference_data_df[0]) != len(columns_names):
                    raise ValueError(
                        "Number of columns in the inference feature is not equal to the number of column names provided")
                else:
                    inference_data_df = pd.DataFrame(data=inference_data_df, columns=columns_names)
            else:
                inference_data_df = pd.DataFrame(data=inference_data_df)

        # checking and converting to DataFrame if the target is in numpy array format
        if type(inference_score_df) is np.ndarray:
            inference_score_df = pd.DataFrame(data=inference_score_df)

        # Checking and converting to DataFrame if the target is in list format
        elif isinstance(inference_score_df, list):
            inference_score_df = pd.DataFrame(data=inference_score_df)

        # checking if the observation/subject numbers are equal in features and in targets
        if inference_data_df.shape[0] != inference_score_df.shape[0]:
            raise ValueError(
                "Number of rows in the inference feature is not equal to the number of rows in inference score")

    if fixed_features is not None:
        if isinstance(fixed_features, list):
            for feature in fixed_features:
                if feature not in feature_df.columns:
                    raise ValueError(
                        f'You provided a fixed feature "{feature}," which does not exist in the Feature matrix.')
            specialist_features = feature_df[fixed_features]
            feature_df = feature_df.drop(fixed_features, axis=1)
        elif isinstance(fixed_features, pd.DataFrame):
            if feature_df.shape[0] != fixed_features.shape[0]:
                raise ValueError(
                    "Number of rows in the Feature is not equal to the number of rows in the fixed features.")
            specialist_features = copy.deepcopy(fixed_features)
        elif type(fixed_features) is np.ndarray:
            specialist_features = pd.DataFrame(data=fixed_features)
            if feature_df.shape[0] != specialist_features.shape[0]:
                raise ValueError(
                    "Number of rows in the Feature is not equal to the number of rows in the fixed features.")
        else:
            raise ValueError(
                "Fixed features must be provided either as a List of feature names, or numpy array, or as a DataFrame containing features.")
    else:
        specialist_features = []

    # Assigning a task type if None
    if task_type == None:
        task_type = 'regression'

    if task_type == 'regression' and probability == True:
        raise ValueError("Probability values cannot be generated for regression tasks.")

    # Checking preconditions for balancing imbalanced data
    if balance == True and task_type == 'regression':
        raise ValueError("Data cannot be made balanced for regression task.")
    elif balance == True and score_df.nunique()[0] > 2:
        raise ValueError(
            "Balancing data only works for binary classification problem. You have more than 2 classes.")

    # Assigning a model if None, or, assigning models based on task type
    if task_type == 'regression':
        if model_name == None or model_name == 'linearSVR':
            if model_kwargs is None:
                model_kwargs = {
                    'C': 1.0,
                    'epsilon': 0.2,
                }

            model = SVR(kernel='linear', **model_kwargs)  # Default
        elif model_name == 'gaussianSVR':
            if model_kwargs is None:
                model_kwargs = {
                    'C': 1.0,
                    'gamma': 'scale',
                }
            model = SVR(kernel='rbf', **model_kwargs)
        elif model_name == 'RegressionForest':
            if model_kwargs is None:
                model_kwargs = {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 5,
                }
            model = RandomForestRegressor(**model_kwargs)
        elif model_name == 'AdaBoostDT':
            if model_kwargs is None:
                model_kwargs = {
                    'estimator': None,
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'random_state': 42,
                }

            model = AdaBoostRegressor(**model_kwargs)
        elif model_name == 'AdaBoostSVR':
            if model_kwargs is None:
                model_kwargs = {
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'random_state': 42,
                }
            model = AdaBoostRegressor(estimator=SVR(), **model_kwargs)
        elif model_name == 'consensus':
            if model_kwargs is None:
                model_kwargs = {
                    'Regression Forest': {'n_estimators': 100, 'random_state': 42, 'max_depth': 5},
                    'Gaussian SVR': {'C': 1.0, 'gamma': 'scale'},
                    'Linear SVR': {'C': 1.0, 'epsilon': 0.2},
                }
            model = {
                "Regression Forest": RandomForestRegressor(**model_kwargs['Regression Forest']),
                "Gaussian SVR": SVR(kernel='rbf', **model_kwargs['Gaussian SVR']),
                "Linear SVR": SVR(kernel='linear', **model_kwargs['Linear SVR'])
            }
        else:
            raise ValueError(
                "Unknown model name. You might have misspelled the name or chose a model that does classification by mistake.")
    elif task_type == 'classification':
        if model_name == None or model_name == 'linearSVC':
            if model_kwargs is None:
                model_kwargs = {'C': 1.0}
            model = SVC(kernel='linear', **model_kwargs)  # Default
            if probability == True:
                model_infer = SVC(kernel='linear', probability=True, **model_kwargs)
        elif model_name == 'gaussianSVC':
            if model_kwargs is None:
                model_kwargs = {'C': 1.0, 'gamma': 'scale'}
            model = SVC(kernel='rbf', **model_kwargs)
            if probability == True:
                model_infer = SVC(kernel='rbf', probability=True, **model_kwargs)
        elif model_name == 'RandomForest':
            if model_kwargs is None:
                model_kwargs = {'n_estimators': 100, 'random_state': 42, 'max_depth': 5}
            model = RandomForestClassifier(**model_kwargs)
        elif model_name == 'AdaBoostDT':
            if model_kwargs is None:
                model_kwargs = {
                    'estimator': None,
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'random_state': 42,
                }
            model = AdaBoostClassifier(**model_kwargs)
        elif model_name == 'AdaBoostSVC':
            if model_kwargs is None:
                model_kwargs = {
                    'algorithm': 'SAMME',
                    'n_estimators': 50,
                    'learning_rate': 1.0,
                    'random_state': 42,
                }
            model = AdaBoostClassifier(estimator=SVC(), **model_kwargs)
        elif model_name == 'consensus':
            if model_kwargs is None:
                model_kwargs = {
                    'Random Forest': {'n_estimators': 100, 'random_state': 42, 'max_depth': 5},
                    'Gaussian SVC': {'C': 1.0, 'gamma': 'scale'},
                    'Linear SVC': {'C': 1.0},
                }
            model = {
                "Random Forest": RandomForestClassifier(**model_kwargs['Random Forest']),
                "Gaussian SVC": SVC(kernel='rbf', **model_kwargs['Gaussian SVC']),
                "Linear SVC": SVC(kernel='linear', **model_kwargs['Linear SVC'])
            }
            if probability == True:
                model_infer = {
                    "Random Forest": RandomForestClassifier(**model_kwargs['Random Forest']),
                    "Gaussian SVC": SVC(kernel='rbf', probability=True, **model_kwargs['Gaussian SVC']),
                    "Linear SVC": SVC(kernel='linear', probability=True, **model_kwargs['Linear SVC'])
                }
        else:
            raise ValueError(
                "Unknown model name. You might have misspelled the name or chose a model that does regression by mistake.")
    else:
        raise ValueError("Unknown task. Please select either 'regression' or 'classification.'")

    if task_type == 'regression' and metric == None:
        metric = 'MAE'  # Default
    elif task_type == 'classification' and metric == None:
        metric = 'Accuracy'   # Default

    if task_type == 'regression':
        if metric == 'Accuracy' or metric == 'binaryROC' or metric == 'F1-score':
            raise ValueError(
                f'{metric} is not supported for regression task. Choose from MAP or MAPE')
    elif task_type == 'classification':
        if metric == 'MAE' or metric == 'MAPE':
            raise ValueError(
                f'{metric} is not supported for classification task. Choose from Accuracy, binaryROC, or F1-score.')

    if voting_strictness == None:
        voting_strictness = 'weighted'  # Default

    if n_fold == None:
        n_fold = 5   # Default

    if max_iter == None:
        max_iter = 3  # Default
    elif max_iter < 1:
        raise ValueError("maxIter must be a positive integer and greater than zero.")

    vote = 0
    if voting_strictness == 'strict':
        vote = round(0.6 * n_fold)
    elif voting_strictness == 'loose':
        vote = round(0.4 * n_fold)
    elif voting_strictness == 'conditional':
        pass
    elif voting_strictness == 'weighted':
        pass
    elif voting_strictness == 'union':
        pass
    else:
        raise ValueError("Unknown voting strictness. Must be either 'strict' or 'loose.'")

    if tolerance == None:
        tolerance = 0.05  # Default
    elif tolerance == 1:
        raise ValueError("tolerance cannot be 1.")

    if max_features == None:
        max_features = round(0.25 * feature_df.shape[1])  # Default
    elif max_features > 1:
        raise ValueError(
            "maxFeatures cannot be greater than 1, i.e., the number of features available.")
    else:
        max_features = round(max_features * feature_df.shape[1])

    if save_intermediate and output_dir is None:
        raise ValueError("Directory for saving intermediate results is not provided.")

    flag_union = 0
    shuffle_flag = True
    if shuffle_flag == False:
        random_seed = None
    else:
        random_seed = 99  # default

    # training a model
    selected_features = train.train(max_iter, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features,
                                    task_type, balance, model_name, model, metric, tolerance, max_features, save_intermediate, output_dir, verbose)

    # selectedFeatures = [[features in fold-1], [features in fold-2],...., [features in fold-n]]

    print(f"\n============================== Inference =====================================\n")

    dfw = None

    # inference
    if probability == True:
        model = model_infer

    if voting_strictness == 'strict' or voting_strictness == 'loose':
        X = [item for sublist in selected_features for item in sublist]
        selected_features = Counter(X)
        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}\n")
        final_features = [element for element, count in selected_features.items() if count >= vote]
        subject_list, actual_score, predicted_score, validationPerformance = inference(
            final_features, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features

    elif voting_strictness == 'weighted':  # weighted #default
        # making sure that features from strict voting are always included
        X = [item for sublist in selected_features for item in sublist]
        Feature_counts = Counter(X)
        # Keeping the order of features (descending) based on their frequency
        Feature_counts_sorted = sorted(Feature_counts.items(), key=lambda x: x[1], reverse=True)

        strict_voting_features = [item[0] for item in Feature_counts_sorted if item[1] >= 3]
        print(f"Features selected atleast 3 times that will be included : {strict_voting_features}")

        max_length = max(len(sublist) for sublist in selected_features)  # max_length = Km
        dict_list = []
        for sublist in selected_features:
            length = len(sublist)
            dict_list.append({sublist[i]: max_length - i for i in range(length)})

        if verbose:
            print(f"Features with assigned ranks in each {n_fold}-folds: {dict_list}\n")

        final_dict = {}
        for d in dict_list:
            for key, value in d.items():
                if key in final_dict:
                    final_dict[key] += value
                else:
                    final_dict[key] = value

        if verbose:
            print(f"Final feature set over {n_fold}-folds with weighted ranks: {final_dict}\n")

        threshold = max_length  # Km
        final_dict_sorted = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)

        weighted_features_with_rank = [item for item in final_dict_sorted if item[1] >= threshold]
        print(
            f"Features selected after thresholding) the weighted features, threshold = {threshold} : {weighted_features_with_rank}")
        weighted_features = [item[0] for item in final_dict_sorted if item[1] >= threshold]

        for _ in strict_voting_features:
            if _ not in weighted_features:
                weighted_features.append(_)
        final_features = copy.deepcopy(weighted_features)

        if verbose:
            print(
                f"Features selected that satisfied the threshold of {threshold} including features selected at least 3 times: {final_features}\n")

        subject_list, actual_score, predicted_score, validationPerformance = inference.inference(
            final_features, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features

        # --------- Estimation of Weights in Percentage -------
        data = final_dict
        data_final = final_features

        # Filter data using data_final
        filtered_data = {key: value for key, value in data.items() if key in data_final}

        # Create DataFrame
        dfw = pd.DataFrame(filtered_data.items(), columns=["Feature", "Weight"])

        # Calculate relative weights as a percentage
        total_weight = dfw["Weight"].sum()
        dfw["Relative Weight (%)"] = (dfw["Weight"] / total_weight) * 100

        # Sort by relative weight for the plot
        dfw = dfw.sort_values("Relative Weight (%)", ascending=False)

        # -----------------------------------------------------

    elif voting_strictness == 'union':  # union
        X = [item for sublist in selected_features for item in sublist]
        Features_counts = Counter(X)
        # Keeping the order of features based on their frequency
        selected_features = sorted(Features_counts.items(), key=lambda x: x[1], reverse=True)
        final_features = [item[0] for item in selected_features]
        if verbose:
            print(f"Following features appeared in the union: {final_features}\n")
        subject_list, actual_score, predicted_score, validationPerformance = inference(
            final_features, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features

    elif voting_strictness == 'conditional':
        X = [item for sublist in selected_features for item in sublist]
        selected_features = Counter(X)

        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}")
        total_sum = sum(selected_features.values())
        mean_value = total_sum / n_fold
        floored_mean = math.floor(mean_value)
        if verbose:
            print(f"Average number of features over 5-folds: {floored_mean}")

        # for strict choice
        final_features = [element for element, count in selected_features.items()
                          if count >= round(0.6 * n_fold)]
        if len(final_features) >= (2 / 3) * floored_mean:
            if verbose:
                print(
                    f"Number of features (w/o specialist features) after strict voting: {len(final_features)}, which is greater or equal 2/3 of the average number of features (i.e., {floored_mean}) over {n_fold}-folds. So keeping 'strict' features.\n")
            subject_list, actual_score, predicted_score, validationPerformance = inference(
                final_features, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
        else:
            final_features = [element for element,
                              count in selected_features.items() if count >= round(0.4 * n_fold)]
            # union
            if len(final_features) <= 2 and floored_mean > 4:
                if verbose:
                    print(
                        f"Number of features (w/o specialist features) after loose voting: {len(final_features)}, which is less or equal 2, while the average number of features (i.e., {floored_mean}) over {n_fold}-folds is greater than 4. So keeping 'union' of features.\n")
                union_list = list(selected_features.keys())
                subject_list, actual_score, predicted_score, validationPerformance = inference(
                    union_list, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
                if len(specialist_features) != 0:
                    final_features = list(specialist_features.columns) + union_list
                else:
                    final_features = union_list
            else:
                # for loose choice
                if verbose:
                    print(
                        f"Number of features (w/o specialist features) after loose voting: {len(final_features)}, which is greater than 2. So keeping 'loose' features.\n")
                subject_list, actual_score, predicted_score, validationPerformance = inference(
                    final_features, n_fold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
                if len(specialist_features) != 0:
                    final_features = list(specialist_features.columns) + final_features

    # inference on additional data
    if inference_data_df is not None and inference_score_df is not None:
        X = [item for sublist in selected_features for item in sublist]
        selected_features = Counter(X)

        if voting_strictness == 'strict' or voting_strictness == 'loose':
            final_features = [element for element,
                              count in selected_features.items() if count >= vote]
            subject_list_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional.inference_additional(
                final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features

        elif voting_strictness == 'conditional':  # conditional
            # for strict choice
            final_features = [element for element,
                              count in selected_features.items() if count >= round(0.6 * n_fold)]
            if len(final_features) >= (2 / 3) * floored_mean:
                subject_list_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional.inference_additional(
                    final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
            else:
                final_features = [element for element,
                                  count in selected_features.items() if count >= round(0.4 * n_fold)]
                # union
                if len(final_features) <= 2 and floored_mean > 4:
                    union_list = list(selected_features.keys())
                    subject_list_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional.inference_additional(
                        final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
                else:
                    # for loose choice
                    subject_list_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional.inference_additional(
                        final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type

        actual_score += [actual_score_add]
        predicted_score += [predicted_score_add]
        validationPerformance += [validationPerformance_add]

    return final_features, subject_list, actual_score, predicted_score, validationPerformance, dfw
