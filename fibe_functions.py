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
# Last Updated: 11/04/2025 at 1200H EST, By Mohammmad Arafat Hussain.


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import copy
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.utils import resample
import os
from data_curation import data_curation, log_files_generator
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import math
import sklearn

sk_version = int(sklearn.__version__.split('.')[0])

def fibe(feature_df, score_df, data_cleaning=False, fixed_features=None, columns_names=None, task_type=None, probability=False, balance=False, model_name=None, metric=None, voting_strictness=None, nFold=None, maxIter=None, tolerance=None, maxFeatures=None, save_intermediate=False, output_dir=None, inference_data_df=None, inference_score_df=None, verbose=True):
    
    '''
    feature_df: is the 2D feature matrix (supports DataFrame, Numpy Array, and List) with columns representing different features.
    score_df: is the 1D score vector as a column (supports DataFrame, Numpy Array, and List).
    data_cleaning: if True, cleans the data including dropping invalid and imbalanced features, mapping categories to numeric values, imputing data with median/mean values.
    fixed_features: Predefined features that must stay in the feature set and the FIBE algorithm does not add or remove those. 
            Must be either a List of names to select from 'feature_df', or DataFrame of features added separately to 'feature_df.'
    columns_names: contain the names of the features. The algorithm returns the names of the selected features from this list. 
            If not available, then the algorithm returns the column indexes of selected features. 
    task_type: either 'regression' or 'classification.' The default is 'regression.'
    probability: if True, probability values (for the class 1) that leads to binary classifications is returned. This option only works when the 'task_type' is 'classification'.
    balance: In a binary classification task, if the data is imbalanced in terms of classes, 'balance=True' uses resampling to balance the data.
    model_name: For the 'regression' task, choose from 'linearSVR', 'gaussianSVR', 'RegressionForest', 'AdaBoostDT', 'AdaBoostSVR', 
            and 'consensus' (consensus using 'linearSVR', 'gaussianSVR', and 'RegressionForest'). The default is 'linearSVR'. For 'classification' task, 
            to choose from 'linearSVC', 'gaussianSVC', 'RandomForest', 'AdaBoostDT', 'AdaBoostSVC', and 'consensus' (consensus using 'linerSVC', 
            'gaussianSVC', and 'RandomForest'). The default is 'linearSVC'.
    metric: For the 'regression' task, choose from 'MAE' and 'MAPE'. The default is 'MAE.' For the 'classification' task, choose from 'Accuracy', 
            'F1-score,' and 'binaryROC'. The default is 'Accuracy'.
    voting_strictness: Choose from 'strict', 'loose', 'weighted', 'union', 'conditional', '2-stage-selection-with-union', '2-stage-selection-with-weighted-voting', or 'best-fold'. The default is 'weighted'.
            'strict': chooses those features that are selected at least 0.6 X N times in N-fold cross-validation.
            'loose': chooses those features that are selected at least 0.4 X N times in N-fold cross-validation.
            'weighted': uses weighted ranking based on feature positions in each fold's selected feature list, with threshold at max_length (Km), and ensures features selected >=3 times (strict) are included.
            'union': takes the union of all features selected across all N outer folds.
            'conditional': first tries strict voting, then falls back to loose voting, and finally to union based on specific conditions.
            '2-stage-selection-with-union': first stage takes union of features from N outer folds, then reruns the entire FIBE process on these features with reshuffled data partitions (different random seed) to produce a second set of N feature selections, and finally takes union of the second stage features as the final selection.
            '2-stage-selection-with-weighted-voting': first stage takes union of features from N outer folds, reruns the FIBE process on these features with reshuffled data partitions, and finally applies weighted majority voting across all selected feature sets from both stages (total 2 x N lists) to determine the final feature subset.
            'best-fold': evaluates each outer fold's selected features on all other (N-1) outer folds using N inner folds cross-validation on each, computes mean performance (accuracy/error) for each fold, and selects the fold with best mean performance (highest for classification, lowest for regression) as the final feature set.
    nFold: Number of folds in cross-validation. Preferred and default is '5'.
    maxIter: is the maximum number of iterations that the algorithm goes back and forth in forward inclusion and backward elimination in each fold. The default is '3'.
    tolerance: is the percentage of deviation in the error/accuracy threshold allowed. The default is '0.05', i.e., 5%.
    maxFeatures: is the number that indicate the number of features to be allowed under tolerance. Default is '3'.
    save_intermediate: if True, saves intermediate results to the specified directory. Default is False.
    output_dir: directory where intermediate results are saved if save_intermediate is True.
    inference_data_df: data for optional second inference cohort for prediction using the selected subset of features.
    inference_score_df: scores for optional second inference cohort for prediction using the selected subset of features.
    verbose: generates text for intermediate loss and selected feature list during iteration. The default is 'True'.

    The outputs are (in order):
    final_features: is the list of features if 'columns_names' was not 'None'. Otherwise column indexes of the selected features. 
            This represents the final selected feature set after applying the chosen voting_strictness method.
    subjectList: is the list for subjects used in inference. Each subject/patient is assigned a name as 'subXX' and according to this list, 
            other outputs are organized in the subsequent generated lists.
    actualScore: is the list containing actual target scores. If 'model_name' is chosen as 'consensus', this list has a repetition of values 3 times, 
            to correspond to predictions by three models. If the argument 'save_intermediate' is set 'True', 'actualScore[-1]' contains an additional 
            list of actual score values of the inference data (if inference_data_df is provided).
    predictedScore: is the list containing predicted scores. If 'model_name' is chosen as 'consensus', this list has 3 predictions per observation. 
            Although 3 predictions per observation are generated here, 'consensus' uses an averaging of the losses for 3 predictions in decision-making. 
            If the argument 'probability' is set 'True' and 'task_type' is 'classification', then predictedScore contains an additional list of 
            prediction probability for class 1 score values for the inference data. The structure is then [predicted, predicted_probs].
    validationPerformance: is a list containing validation performance in terms of chosen 'metric' for 'nFold' folds. Each element corresponds to 
            the performance on one fold during cross-validation inference.
    dfw: is a DataFrame containing feature weights (for 'weighted' voting_strictness) or None (for other voting methods). When available, it contains 
            columns: 'Feature', 'Weight', and 'Relative Weight (%)' sorted by relative weight in descending order. 
    '''
    
    start_time = datetime.now()
    print("Code started running at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def apply_weighted_voting(selected_features_lists, verbose=False, descriptor=None):
        if not selected_features_lists:
            return [], pd.DataFrame(columns=["Feature", "Weight", "Relative Weight (%)"])

        num_lists = len(selected_features_lists)
        descriptor = descriptor or f"{num_lists} folds"

        feature_counts = Counter([item for sublist in selected_features_lists for item in sublist])
        strict_threshold = max(1, round(0.6 * num_lists))
        strict_voting_features = [feat for feat, count in feature_counts.items() if count >= strict_threshold]
        if verbose:
            print(f"Features selected at least {strict_threshold} times that will be included : {strict_voting_features}")

        max_length = max((len(sublist) for sublist in selected_features_lists), default=0)
        dict_list = []
        for sublist in selected_features_lists:
            length = len(sublist)
            dict_list.append({sublist[i]: max_length - i for i in range(length)})

        if verbose:
            print(f"Features with assigned ranks in each set ({descriptor}): {dict_list}\n")

        final_dict = {}
        for d in dict_list:
            for key, value in d.items():
                final_dict[key] = final_dict.get(key, 0) + value

        if verbose:
            print(f"Final feature set over {descriptor} with weighted ranks: {final_dict}\n")

        threshold = max_length
        final_dict_sorted = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)

        if verbose:
            filtered_sorted = [(item[0], item[1]) for item in final_dict_sorted if item[1] >= threshold]
            print(f"Features selected after thresholding the weighted features, threshold = {threshold} : {filtered_sorted}")

        weighted_features = [item[0] for item in final_dict_sorted if item[1] >= threshold]

        for feat in strict_voting_features:
            if feat not in weighted_features:
                weighted_features.append(feat)

        if verbose:
            print(f"Features selected that satisfied the threshold of {threshold} including features selected at least {strict_threshold} times: {weighted_features}\n")

        if weighted_features:
            filtered_data = {key: value for key, value in final_dict.items() if key in weighted_features}
            dfw_local = pd.DataFrame(filtered_data.items(), columns=["Feature", "Weight"])
            total_weight = dfw_local["Weight"].sum()
            if total_weight == 0:
                dfw_local["Relative Weight (%)"] = 0.0
            else:
                dfw_local["Relative Weight (%)"] = (dfw_local["Weight"] / total_weight) * 100
            dfw_local = dfw_local.sort_values("Relative Weight (%)", ascending=False)
        else:
            dfw_local = pd.DataFrame(columns=["Feature", "Weight", "Relative Weight (%)"])

        return weighted_features, dfw_local
    
    # Data Curation : Added by Ankush Kesri
    if data_cleaning == True:
        feature_df, drop_log, mapping_list, imputation_log = data_curation(feature_df)
    
        # Saving log files
        log_files_generator(drop_log, mapping_list, imputation_log, output_dir)
    
    
    # Checking and converting to DataFrame if the features are in numpy array format
    if type(feature_df) is np.ndarray:
        if columns_names != None:
            # checking if the length of feature names is equal to the feature column numbers
            if feature_df.shape[1] != len(columns_names):
                raise ValueError("Number of columns in the Feature is not equal to the number of column names provided")
            else:
                feature_df = pd.DataFrame(data=feature_df, columns=columns_names)
        else:
            feature_df = pd.DataFrame(data=feature_df)
            
    # Checking and converting to DataFrame if the features are in list format
    elif isinstance(feature_df, list):
        if columns_names != None:
            # checking if the length of feature names are equal to the feature column numbers
            if len(feature_df[0]) != len(columns_names):
                raise ValueError("Number of columns in the Feature is not equal to the number of column names provided")
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
        raise ValueError("Number of rows in the Feature is not equal to the number of rows in Score")

    if inference_data_df is not None and inference_score_df is None:
        raise ValueError("Scores for inference cohort is not provided.")
    
    if inference_data_df is not None and inference_score_df is not None:
        # Checking and converting to DataFrame if the inference features are in numpy array format
        if type(inference_data_df) is np.ndarray:
            if columns_names != None:
                # checking if the length of feature names is equal to the feature column numbers
                if inference_data_df.shape[1] != len(columns_names):
                    raise ValueError("Number of columns in the inference feature is not equal to the number of column names provided")
                else:
                    inference_data_df = pd.DataFrame(data=inference_data_df, columns=columns_names)
            else:
                inference_data_df = pd.DataFrame(data=inference_data_df)
                
        # Checking and converting to DataFrame if the features are in list format
        elif isinstance(inference_data_df, list):
            if columns_names != None:
                # checking if the length of feature names are equal to the feature column numbers
                if len(inference_data_df[0]) != len(columns_names):
                    raise ValueError("Number of columns in the inference feature is not equal to the number of column names provided")
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
            raise ValueError("Number of rows in the inference feature is not equal to the number of rows in inference score")
    
    if fixed_features is not None:
        if isinstance(fixed_features, list):
            for feature in fixed_features: 
                if feature not in feature_df.columns:
                    raise ValueError(f'You provided a fixed feature "{feature}," which does not exist in the Feature matrix.')
            specialist_features = feature_df[fixed_features]
            feature_df = feature_df.drop(fixed_features, axis=1)
        elif isinstance(fixed_features, pd.DataFrame):
            if feature_df.shape[0] != fixed_features.shape[0]:
                raise ValueError("Number of rows in the Feature is not equal to the number of rows in the fixed features.")
            specialist_features = copy.deepcopy(fixed_features)
        elif type(fixed_features) is np.ndarray:
            specialist_features = pd.DataFrame(data=fixed_features)
            if feature_df.shape[0] != specialist_features.shape[0]:
                raise ValueError("Number of rows in the Feature is not equal to the number of rows in the fixed features.")
        else:
            raise ValueError("Fixed features must be provided either as a List of feature names, or numpy array, or as a DataFrame containing features.")
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
        raise ValueError("Balancing data only works for binary classification problem. You have more than 2 classes.")
    
    # Assigning a model if None, or, assigning models based on task type
    if task_type == 'regression':
        if model_name == None or model_name == 'linearSVR':
            model = SVR(kernel = 'linear', C=1.0, epsilon=0.2)  # Default
        elif model_name == 'gaussianSVR':
            model = SVR(kernel = 'rbf', C=1.0, gamma='scale')
        elif model_name == 'RegressionForest':
            model = RandomForestRegressor(n_estimators = 100, random_state=42, max_depth=5)
        elif model_name == 'AdaBoostDT':
            if sk_version == 0:
                model = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
            else:
                model = AdaBoostRegressor(estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'AdaBoostSVR':
            if sk_version == 0:
                model = AdaBoostRegressor(base_estimator=SVR(), n_estimators=50, learning_rate=1.0, random_state=42)
            else:
                model = AdaBoostRegressor(estimator=SVR(), n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'consensus':
            model = { 
            "Regression Forest" : RandomForestRegressor(n_estimators = 100, random_state=42, max_depth=5),
            "Gaussian SVR" : SVR(kernel = 'rbf', C=1.0, gamma='scale'), 
            "Linear SVR" : SVR(kernel = 'linear', C=1.0, epsilon=0.2)
            } 
        else:
            raise ValueError("Unknown model name. You might have misspelled the name or chose a model that does classification by mistake.")
    elif task_type == 'classification':
        if model_name == None or model_name == 'linearSVC':
            model = SVC(kernel = 'linear', C=1.0)  # Default
            if probability == True:
                model_infer = SVC(kernel = 'linear', C=1.0, probability=True) 
        elif model_name == 'gaussianSVC':
            model = SVC(kernel = 'rbf', C=1.0, gamma='scale')
            if probability == True:
                model_infer = SVC(kernel = 'rbf', C=1.0, gamma='scale', probability=True)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        elif model_name == 'AdaBoostDT':
            if sk_version == 0:
                model = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
            else:
                model = AdaBoostClassifier(estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'AdaBoostSVC':
            if sk_version == 0:
                model = AdaBoostClassifier(base_estimator=SVC(), algorithm='SAMME', n_estimators=50, learning_rate=1.0, random_state=42)
            else:
                model = AdaBoostClassifier(estimator=SVC(), algorithm='SAMME', n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'consensus':
            model = { 
                "Random Forest" : RandomForestClassifier(n_estimators = 100, random_state=42, max_depth=5),
                "Gaussian SVC" : SVC(kernel = 'rbf', C=1.0, gamma='scale'), 
                "Linear SVC" : SVC(kernel = 'linear', C=1.0)
                }
            if probability == True:
                model_infer = { 
                "Random Forest" : RandomForestClassifier(n_estimators = 100, random_state=42, max_depth=5),
                "Gaussian SVC" : SVC(kernel = 'rbf', C=1.0, gamma='scale', probability=True), 
                "Linear SVC" : SVC(kernel = 'linear', C=1.0, probability=True)
                }     
        else:
            raise ValueError("Unknown model name. You might have misspelled the name or chose a model that does regression by mistake.")
    else:
        raise ValueError("Unknown task. Please select either 'regression' or 'classification.'")
        
    if task_type == 'regression' and metric == None:
        metric = 'MAE' # Default 
    elif task_type == 'classification' and metric == None:
        metric = 'Accuracy'   # Default 
        
    if task_type == 'regression':
        if metric == 'Accuracy' or  metric == 'binaryROC' or metric == 'F1-score':
            raise ValueError(f'{metric} is not supported for regression task. Choose from MAP or MAPE')
    elif task_type == 'classification':
        if metric == 'MAE' or  metric == 'MAPE':
            raise ValueError(f'{metric} is not supported for classification task. Choose from Accuracy, binaryROC, or F1-score.')
    
    if voting_strictness == None:
        voting_strictness = 'weighted'  # Default 
        
    if nFold == None:
        nFold = 5   # Default 
        
    if maxIter == None:
        maxIter = 3  # Default 
    elif maxIter < 1:
        raise ValueError("maxIter must be a positive integer and greater than zero.")
        
    if voting_strictness == 'strict':
        vote = round(0.6 * nFold)
    elif voting_strictness == 'loose':
        vote = round(0.4 * nFold)
    elif voting_strictness == 'conditional':
        vote = 101
    elif voting_strictness == 'weighted':
        vote = 102
    elif voting_strictness == 'union':
        vote = 103
    elif voting_strictness == '2-stage-selection-with-union':
        vote = 104
    elif voting_strictness == 'best-fold':
        vote = 105
    elif voting_strictness == '2-stage-selection-with-weighted-voting':
        vote = 106
    else:
        raise ValueError("Unknown voting strictness. Must be either 'strict', 'loose', 'weighted', 'union', 'conditional', '2-stage-selection-with-union', '2-stage-selection-with-weighted-voting', or 'best-fold.'")
        
    if tolerance == None:
        tolerance = 0.05  # Default 
    elif tolerance == 1:
        raise ValueError("tolerance cannot be 1.")
        
    if maxFeatures == None:
        maxFeatures = 3  # Default  previously --> round(0.25*feature_df.shape[1])
    elif maxFeatures > feature_df.shape[1]:
        raise ValueError("maxFeatures cannot be greater than the total number of features available.")
    
    if save_intermediate and output_dir is None:
        raise ValueError("Directory for saving intermediate results is not provided.")


    flag_union = 0
    shuffle_flag = True
    if shuffle_flag == False:
        random_seed = None
    else:
        random_seed = 99  #default
    
    # training a model
    selectedFeatures = train(maxIter, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, task_type, balance, model_name, model, metric, tolerance, maxFeatures, save_intermediate, output_dir, verbose)
    
    # selectedFeatures = [[features in fold-1], [features in fold-2],...., [features in fold-n]]
    
    # 2-stage-selection variations
    dfw_stage_combined = None
    if vote in (104, 106):  # two-stage selections
        if verbose:
            print(f"\n============================== Stage 1 Complete =====================================\n")
            print(f"Running {voting_strictness} approach...\n")
        
        selectedFeatures_stage1 = copy.deepcopy(selectedFeatures)
        
        # Stage 1: Get union of all features from first stage
        X_stage1 = [item for sublist in selectedFeatures_stage1 for item in sublist]
        stage1_union = list(set(X_stage1))  # Get unique features (union)
        
        if verbose:
            print(f"Stage 1: Union of features from {nFold} outer folds: {stage1_union}\n")
            print(f"Number of features selected in Stage 1: {len(stage1_union)}\n")
        
        # Filter feature_df to only include features from stage 1 union
        if len(stage1_union) == 0:
            raise ValueError("Stage 1 did not select any features. Cannot proceed to Stage 2.")
        
        feature_df_stage2 = feature_df[stage1_union].copy()
        
        if verbose:
            print(f"\n============================== Stage 2: Training on Stage 1 Features =====================================\n")
        
        # Stage 2: Use different random seed to reshuffle data partitions
        if random_seed is None:
            random_seed_stage2 = None
        else:
            random_seed_stage2 = random_seed + 1000
        
        if verbose:
            print(f"Stage 2: Using random seed {random_seed_stage2} (different from Stage 1 seed {random_seed}) for reshuffled data partitions.\n")
        
        # Training stage 2 on filtered features with new seed
        selectedFeatures_stage2 = train(maxIter, nFold, feature_df_stage2, score_df, shuffle_flag, random_seed_stage2, specialist_features, task_type, balance, model_name, model, metric, tolerance, maxFeatures, save_intermediate, output_dir, verbose)
        
        # Stage 2: Get union of all features from second stage
        X_stage2 = [item for sublist in selectedFeatures_stage2 for item in sublist]
        final_features_stage2 = list(set(X_stage2))  # Get unique features (union)
        
        if verbose:
            print(f"\n============================== Stage 2 Complete =====================================\n")
            print(f"Stage 2: Union of features from {nFold} outer folds: {final_features_stage2}\n")
            print(f"Number of features selected in Stage 2: {len(final_features_stage2)}\n")
        
        selectedFeatures = copy.deepcopy(selectedFeatures_stage2)
        
        if vote == 104:  # with union
            final_features = final_features_stage2.copy()
        elif vote == 106:  # with weighted voting across both stages
            combined_feature_lists = selectedFeatures_stage1 + selectedFeatures_stage2
            combined_descriptor = f"{len(combined_feature_lists)} fold sets (Stage 1 + Stage 2)"
            final_features_weighted, dfw_stage_combined = apply_weighted_voting(combined_feature_lists, verbose=verbose, descriptor=combined_descriptor)
            final_features = copy.deepcopy(final_features_weighted)
        
    print(f"\n============================== Inference =====================================\n")
    
    # inference
    if probability == True:
        model = model_infer
    
    # Initialize dfw to None (will be set for weighted voting)
    dfw = None
    if vote == 106 and dfw_stage_combined is not None:
        dfw = dfw_stage_combined
        
    if vote == round(0.6 * nFold) or vote == round(0.4 * nFold):
        X = [item for sublist in selectedFeatures for item in sublist]
        selectedFeatures = Counter(X)
        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}\n")
        final_features = [element for element, count in selectedFeatures.items() if count >= vote]
        subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features
    
    elif vote == 102: #weighted #default
        final_features, dfw = apply_weighted_voting(selectedFeatures, verbose=verbose, descriptor=f"{nFold}-folds")
        subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features
    
    elif vote == 103:  #union
        X = [item for sublist in selectedFeatures for item in sublist]
        Features_counts = Counter(X)
        selectedFeatures = sorted(Features_counts.items(), key=lambda x: x[1], reverse=True)   #Keeping the order of features based on their frequency
        final_features = [item[0] for item in selectedFeatures]
        if verbose:
            print(f"Following features appeared in the union: {final_features}\n")
        subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features
    
    elif vote == 104:  #2-stage-selection-with-union
        # final_features was already set in the 2-stage block above
        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}\n")
            print(f"Final features from two-stage union: {final_features}\n")
        subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features
    
    elif vote == 105:  #best-fold
        # final_features was already set in the best-fold block above
        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}\n")
            print(f"Final features from best-fold selection: {final_features}\n")
        subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features
    
    elif vote == 106:  #2-stage-selection-with-weighted-voting
        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}\n")
            print(f"Final features from two-stage weighted voting: {final_features}\n")
        subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + final_features
    
    elif vote == 101:
        X = [item for sublist in selectedFeatures for item in sublist]
        selectedFeatures = Counter(X)
        
        if verbose:
            print(f"\nVoting strictness is selected: {voting_strictness}")
        total_sum = sum(selectedFeatures.values())
        mean_value = total_sum / nFold
        floored_mean = math.floor(mean_value)
        if verbose:
            print(f"Average number of features over 5-folds: {floored_mean}")
        
        # for strict choice
        final_features = [element for element, count in selectedFeatures.items() if count >= round(0.6 * nFold)]
        if len(final_features) >= (2/3)*floored_mean:
            if verbose:
                print(f"Number of features (w/o specialist features) after strict voting: {len(final_features)}, which is greater or equal 2/3 of the average number of features (i.e., {floored_mean}) over {nFold}-folds. So keeping 'strict' features.\n")
            subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
        else:
            final_features = [element for element, count in selectedFeatures.items() if count >= round(0.4 * nFold)]
            # union
            if len(final_features) <= 2 and floored_mean > 4:
                if verbose:
                    print(f"Number of features (w/o specialist features) after loose voting: {len(final_features)}, which is less or equal 2, while the average number of features (i.e., {floored_mean}) over {nFold}-folds is greater than 4. So keeping 'union' of features.\n")
                union_list = list(selectedFeatures.keys())
                subjectList, actual_score, predicted_score, validationPerformance = inference(union_list, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
                if len(specialist_features) != 0:
                    final_features = list(specialist_features.columns) + union_list
                else:
                    final_features = union_list
            else: 
                # for loose choice
                if verbose:
                    print(f"Number of features (w/o specialist features) after loose voting: {len(final_features)}, which is greater than 2. So keeping 'loose' features.\n")
                subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
                if len(specialist_features) != 0:
                    final_features = list(specialist_features.columns) + final_features
        
    # inference on additional data
    if inference_data_df is not None and inference_score_df is not None:
        X = [item for sublist in selectedFeatures for item in sublist]
        selectedFeatures = Counter(X)
        
        if vote == round(0.6 * nFold) or vote == round(0.4 * nFold):
            final_features = [element for element, count in selectedFeatures.items() if count >= vote]
            subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
                
        elif vote == 102 or vote == 103 or vote == 104 or vote == 105 or vote == 106:  #weighted, union, two-stage options, or best-fold
            # final_features was already computed in the main inference section above
            subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
                
        elif vote == 101:
            # for strict choice
            final_features = [element for element, count in selectedFeatures.items() if count >= round(0.6 * nFold)]
            if len(final_features) >= (2/3)*floored_mean:
                subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
            else:
                final_features = [element for element, count in selectedFeatures.items() if count >= round(0.4 * nFold)]
                # union
                if len(final_features) <= 2 and floored_mean > 4:
                    union_list = list(selectedFeatures.keys())
                    subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
                else:
                    # for loose choice
                    subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type

        actual_score = actual_score + [actual_score_add]  
        predicted_score = predicted_score + [predicted_score_add]
        validationPerformance = validationPerformance + [validationPerformance_add]

    return final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw
        
        
def train(maxIter, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, task_type, balance, model_name, model, metric, tolerance, maxFeatures, save_intermediate, output_dir, verbose=False):
    max_iter = maxIter
    kf5 = KFold(n_splits = nFold, shuffle=shuffle_flag, random_state=random_seed)
    
    frequency_of_features_selected_all_fold = []

    oF = 0
    
    if verbose:
        print(f"Total Number of Features to Traverse in each fold: {len(feature_df.columns)}")
        print(f"Tolerance is selected: {tolerance}")
        print(f"Maximum number of Features allowed under tolerance, if not best: {maxFeatures}\n")
        
    for outer_fold in kf5.split(feature_df):
        print("\n=================================================================================\n")
        flag_FI = 0
        flag_BE = 0
        
        
        train_val_df = feature_df.iloc[outer_fold[0]]
        train_val_score_df = score_df.iloc[outer_fold[0]]
        if len(specialist_features) != 0:
            train_specialist = specialist_features.iloc[outer_fold[0]]
        
        oF += 1
            
        selected_features = []
        best_features = []
        
        if task_type == 'regression':
            lowest_error = float('inf')
            error_to_fill = lowest_error #newadd
        else:
            highest_accuracy = float('-inf')
            error_to_fill = highest_accuracy #newadd
        
        for i in range(max_iter):
            noFeat_tolerance = 0
            exit_flag = 0
            for q in range(len(feature_df.columns)): 
                if verbose:
                    print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}]", flush=True)
                temp_error = []
                for feature in feature_df.columns:
                    if feature in selected_features:
                        temp_error.append(error_to_fill)
                        continue

                    # Add the new feature to the selected features
                    temp_features = selected_features + [feature]
                    inner_error = []

                    for inner_fold in kf5.split(train_val_df): 
                        training = train_val_df.iloc[inner_fold[0]]
                        validation = train_val_df.iloc[inner_fold[1]]

                        if len(specialist_features) != 0:
                            train_spec = train_specialist.iloc[inner_fold[0]]
                            valid_spec = train_specialist.iloc[inner_fold[1]]

                        # training
                        df_score = train_val_score_df.iloc[inner_fold[0]]
                        df_feature = training[temp_features]

                        # validation
                        df_val_score = train_val_score_df.iloc[inner_fold[1]]
                        df_val_feature = validation[temp_features]

                        if balance == True:
                            if len(specialist_features) != 0:
                                df_feature, df_score = balanceData(pd.concat([train_spec, df_feature], axis=1), df_score)
                                df_val_feature, df_val_score = balanceData(pd.concat([valid_spec, df_val_feature], axis=1), df_val_score)
                            else:
                                df_feature, df_score = balanceData(df_feature, df_score)
                                df_val_feature, df_val_score = balanceData(df_val_feature, df_val_score)
                        else:
                            if len(specialist_features) != 0:
                                df_feature = pd.concat([train_spec, df_feature], axis=1)
                                df_val_feature = pd.concat([valid_spec, df_val_feature], axis=1)

                        if model_name == 'consensus':
                            for one_model in model:
                                model_ = model[one_model]

                                # Train a Regressor with the selected features and predict
                                model_.fit(df_feature, df_score.values.ravel())
                                y_pred = model_.predict(df_val_feature)

                                # Calculate the mean absolute error for the validation set
                                inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                        else:
                            # Train a Regressor with the selected features and predict
                            model.fit(df_feature, df_score.values.ravel())
                            y_pred = model.predict(df_val_feature)

                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation(metric, df_val_score, y_pred))

                    temp_error.append(np.mean(inner_error)) #newadd

                    if verbose:
                        if task_type == 'regression':
                            print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] -> Feature Added: {feature} | Error Found: {np.mean(inner_error)}", flush=True)
                        else:
                            print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] -> Feature Added: {feature} | Accuracy Found: {np.mean(inner_error)}", flush=True)

                if task_type == 'regression':
                    if np.min(temp_error) <= lowest_error*(1+tolerance):
                        selected_features.append(feature_df.columns[np.argmin(temp_error)])
                        
                        if np.min(temp_error) > lowest_error:
                            noFeat_tolerance += 1
                            print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] -- {noFeat_tolerance} feature(s) allowed under tolerance, which is not better than the last best.")
                            if noFeat_tolerance == maxFeatures:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] -- Number of allowed features under tolerance is met. Exiting from further FI | Best Features remain: {best_features}")
                                exit_flag = 1
                                break
                            
                        if np.min(temp_error) < lowest_error:
                            lowest_error = np.min(temp_error)
                            best_features = copy.deepcopy(selected_features)
                            noFeat_tolerance = 0
                             
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Error ({metric}): {lowest_error:.4f} | Current Error ({metric}): {np.min(temp_error):.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Error ({metric}): {lowest_error:.4f} | Current Error ({metric}): {np.min(temp_error):.4f} | Selected Features: {selected_features}", flush=True)
                    elif np.min(temp_error) > lowest_error*(1+tolerance) or exit_flag == 1:
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + best_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] No additional feature improves performance beyond tolerance | Best Features: {all_feat} | Starting BE..", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] No additional feature improves performance beyond tolerance | Best Features: {best_features} | Starting BE..", flush=True)
                        flag_FI = 1
                        break
                else:
                    if np.max(temp_error) >= highest_accuracy*(1-tolerance) and exit_flag == 0:
                        selected_features.append(feature_df.columns[np.argmax(temp_error)])
                        
                        if np.max(temp_error) < highest_accuracy:
                            noFeat_tolerance += 1
                            print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] -- {noFeat_tolerance} feature(s) allowed under tolerance, which is not better than the last best.")
                            if noFeat_tolerance == maxFeatures:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] -- Number of allowed features under tolerance is met. Exiting from further FI | Best Features remain: {best_features}")
                                exit_flag = 1
                                break
                        
                        if np.max(temp_error) > highest_accuracy:
                            highest_accuracy = np.max(temp_error)
                            best_features = copy.deepcopy(selected_features)
                            noFeat_tolerance = 0
                            
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Prediction ({metric}): {highest_accuracy:.4f} | Currenct Prediction ({metric}): {np.max(temp_error):.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Prediction ({metric}): {highest_accuracy:.4f} | Currenct Prediction ({metric}): {np.max(temp_error):.4f} | Selected Features: {selected_features}", flush=True)
                    
                    elif np.max(temp_error) < highest_accuracy*(1-tolerance) or exit_flag == 1: 
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + best_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] No additional feature improves performance beyond tolerance | Best Features: {all_feat} | Starting BE..", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] No additional feature improves performance beyond tolerance | Best Features: {best_features} | Starting BE..", flush=True)
                        flag_FI = 1
                        break
            
            selected_features = copy.deepcopy(best_features)
            if len(selected_features) == 1:
                if verbose:
                    print(f"[Fold: {oF} | Iter: {i+1} | -- | Traversal: -] Since there is only one feature selected in FI, skipping BE and next iterations.", flush=True)
                break 
                
            # Backward Elimination
            #selected_features_ = copy.deepcopy(selected_features)
            
            for q in range(len(selected_features)):
                if verbose:
                    print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] NOTE: You may see less number of features traversed over in BE, missing ones were under tolerance but not the best.", flush=True)
                temp_error = []
                for feature in selected_features:   #changed
                    # remove a feature from the selected features
                    temp_features = copy.deepcopy(selected_features)
                    if len(temp_features) == 1:
                        continue
                    temp_features.remove(feature)
                    inner_error = []

                    for inner_fold in kf5.split(train_val_df): 
                        training = train_val_df.iloc[inner_fold[0]]
                        validation = train_val_df.iloc[inner_fold[1]]

                        if len(specialist_features) != 0:
                            train_spec = train_specialist.iloc[inner_fold[0]]
                            valid_spec = train_specialist.iloc[inner_fold[1]]

                        # training
                        df_score = train_val_score_df.iloc[inner_fold[0]]
                        df_feature = training[temp_features]

                        # validation
                        df_val_score = train_val_score_df.iloc[inner_fold[1]]
                        df_val_feature = validation[temp_features]

                        if balance == True:
                            if len(specialist_features) != 0:
                                df_feature, df_score = balanceData(pd.concat([train_spec, df_feature], axis=1), df_score)
                                df_val_feature, df_val_score = balanceData(pd.concat([valid_spec, df_val_feature], axis=1), df_val_score)
                            else:
                                df_feature, df_score = balanceData(df_feature, df_score)
                                df_val_feature, df_val_score = balanceData(df_val_feature, df_val_score)
                        else:
                            if len(specialist_features) != 0:
                                df_feature = pd.concat([train_spec, df_feature], axis=1)
                                df_val_feature = pd.concat([valid_spec, df_val_feature], axis=1)

                        if model_name == 'consensus':
                            for one_model in model:
                                model_ = model[one_model]

                                # Train a Regressor with the selected features and predict
                                model_.fit(df_feature, df_score.values.ravel())
                                y_pred = model_.predict(df_val_feature)

                                # Calculate the mean absolute error for the validation set
                                inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                        else:
                            # Train a Regressor with the selected features and predict
                            model.fit(df_feature, df_score.values.ravel())
                            y_pred = model.predict(df_val_feature)

                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation(metric, df_val_score, y_pred))

                    temp_error.append(np.mean(inner_error)) #newadd

                    if verbose:
                        if task_type == 'regression':
                            print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] <- Feature Removed: {feature} | Error Found: {np.mean(inner_error)}", flush=True)
                        else:
                            print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] <- Feature Removed: {feature} | Accuracy Found: {np.mean(inner_error)}", flush=True)
            
                if task_type == 'regression':
                    if np.min(temp_error) < lowest_error:
                        lowest_error = np.min(temp_error)                                           
                        selected_features.remove(selected_features[np.argmin(temp_error)])  #changed
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] - Traversal over all features finished | {metric}: {lowest_error:.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] - Traversal over all features finished | {metric}: {lowest_error:.4f} | Selected Features: {selected_features}", flush=True)
                    else:
                        flag_BE = 1
                        if verbose:
                            print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] No removal of additional feature improves performance | Selected Features: {selected_features}", flush=True)
                        break
                else:
                    if np.max(temp_error) > highest_accuracy:
                        highest_accuracy = np.max(temp_error)
                        selected_features.remove(selected_features[np.argmax(temp_error)])  #changed
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] - Traversal over all features finished | {metric}: {highest_accuracy:.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] - Traversal over all features finished | {metric}: {highest_accuracy:.4f} | Selected Features: {selected_features}", flush=True)
                    else:
                        flag_BE = 1
                        if verbose:
                            print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] No removal of additional feature improves performance | Selected Features: {selected_features}", flush=True)
                        break
            
            if flag_FI and flag_BE:
                if verbose:
                    print(f"[Fold: {oF} | Iter: {i+1} | -- | Traversal: -] Since no addition or removal of any features improves performance, skipping next iterations.", flush=True)
                break
            
            if i == 0:
                f_set_1 = copy.deepcopy(selected_features)
            else:
                f_set = copy.deepcopy(selected_features)
                if sorted(f_set_1) == sorted(f_set):
                    if verbose:
                        print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] Selected features in this iteration did not change from the previous iteration. So quiting further iterations.", flush=True)
                    break
                else:
                    f_set_1 = f_set
                    
        # saving intermediate results: features selected in each outer fold
        if save_intermediate:
            if output_dir is not None:
                # Check if the directory exists or can be created
                if not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir)
                    except OSError as e:
                        raise RuntimeError(f"Error creating directory '{output_dir}': {e}")

                # Check if the user has write permission to the directory
                if os.access(output_dir, os.W_OK):
                    # Directory exists and user has write permission
                    with open(os.path.join(output_dir, f"Selected_features_at_fold_{oF}.txt"), "w") as f:
                        f.write(f"{selected_features}")
                else:
                    raise PermissionError(f"You do not have write permission to directory '{output_dir}'")

        # saving features selected across all outer folds
        if oF == 1:
            frequency_of_features_selected_all_fold = [selected_features]
        else:
            frequency_of_features_selected_all_fold = frequency_of_features_selected_all_fold + [selected_features]
        #print(frequency_of_features_selected_all_fold)
    #feature_counts = Counter(frequency_of_features_selected_all_fold)
    #return feature_counts
    return frequency_of_features_selected_all_fold

def evaluate_features_across_folds(selectedFeatures, feature_df, score_df, nFold, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, verbose=False):
    """
    Evaluates each outer fold's selected features on all other (N-1) outer folds.
    For each fold i's features, evaluates on fold j's data using N inner folds.
    Returns the best fold's features along with performance metrics.
    
    selectedFeatures: list of lists, where each inner list contains features selected in one outer fold
    Returns: (best_features, best_fold_idx, all_fold_performances) where all_fold_performances is a list of dicts
             with keys: 'fold_idx', 'mean', 'std', 'features'
    """
    kf5 = KFold(n_splits = nFold, shuffle=shuffle_flag, random_state=random_seed)
    
    # Get all outer fold splits
    outer_fold_splits = list(kf5.split(feature_df))
    
    all_fold_performances = []
    
    if verbose:
        print(f"\n============================== Evaluating Features Across Folds =====================================\n")
        print(f"Evaluating {nFold} outer folds, each on {nFold-1} other folds with {nFold} inner folds each.\n")
    
    # For each outer fold i
    for fold_i_idx in range(nFold):
        selected_features_i = selectedFeatures[fold_i_idx]
        
        if verbose:
            print(f"\nEvaluating Fold {fold_i_idx + 1} features: {selected_features_i}")
        
        performance_list = []
        
        # For each other outer fold j (N-1 folds)
        for fold_j_idx in range(nFold):
            if fold_j_idx == fold_i_idx:
                continue  # Skip the same fold
            
            # Get fold j's data
            fold_j_train_indices = outer_fold_splits[fold_j_idx][0]
            fold_j_data = feature_df.iloc[fold_j_train_indices]
            fold_j_scores = score_df.iloc[fold_j_train_indices]
            
            if len(specialist_features) != 0:
                fold_j_specialist = specialist_features.iloc[fold_j_train_indices]
            
            # Split fold j's data into N inner folds
            kf5_inner = KFold(n_splits=nFold, shuffle=shuffle_flag, random_state=random_seed)
            
            for inner_fold in kf5_inner.split(fold_j_data):
                inner_train_indices = inner_fold[0]
                inner_val_indices = inner_fold[1]
                
                # Training data for this inner fold
                inner_train_data = fold_j_data.iloc[inner_train_indices]
                inner_train_scores = fold_j_scores.iloc[inner_train_indices]
                
                # Validation data for this inner fold
                inner_val_data = fold_j_data.iloc[inner_val_indices]
                inner_val_scores = fold_j_scores.iloc[inner_val_indices]
                
                # Prepare features (selected_features_i from fold i)
                df_feature_train = inner_train_data[selected_features_i]
                df_feature_val = inner_val_data[selected_features_i]
                
                # Handle specialist features
                if len(specialist_features) != 0:
                    inner_train_spec = fold_j_specialist.iloc[inner_train_indices]
                    inner_val_spec = fold_j_specialist.iloc[inner_val_indices]
                    
                    if balance == True:
                        df_feature_train, inner_train_scores = balanceData(pd.concat([inner_train_spec, df_feature_train], axis=1), inner_train_scores)
                        df_feature_val, inner_val_scores = balanceData(pd.concat([inner_val_spec, df_feature_val], axis=1), inner_val_scores)
                    else:
                        df_feature_train = pd.concat([inner_train_spec, df_feature_train], axis=1)
                        df_feature_val = pd.concat([inner_val_spec, df_feature_val], axis=1)
                else:
                    if balance == True:
                        df_feature_train, inner_train_scores = balanceData(df_feature_train, inner_train_scores)
                        df_feature_val, inner_val_scores = balanceData(df_feature_val, inner_val_scores)
                
                # Train and evaluate
                if model_name == 'consensus':
                    predictions = []
                    for one_model in model:
                        model_ = model[one_model]
                        model_.fit(df_feature_train, inner_train_scores.values.ravel())
                        y_pred = model_.predict(df_feature_val)
                        predictions.append(y_pred)
                    
                    if task_type == 'regression':
                        consensus_pred = [(a+b+c)/3 for a,b,c in zip(predictions[0], predictions[1], predictions[2])]
                    elif task_type == 'classification':
                        def majority_vote(a,b,c):
                            return 1 if a+b+c>1 else 0
                        consensus_pred = [majority_vote(a,b,c) for a,b,c in zip(predictions[0], predictions[1], predictions[2])]
                    
                    performance = loss_estimation(metric, inner_val_scores, consensus_pred)
                else:
                    model.fit(df_feature_train, inner_train_scores.values.ravel())
                    y_pred = model.predict(df_feature_val)
                    performance = loss_estimation(metric, inner_val_scores, y_pred)
                
                performance_list.append(performance)
        
        # Calculate mean and std for fold i
        mean_performance = np.mean(performance_list)
        std_performance = np.std(performance_list)
        
        all_fold_performances.append({
            'fold_idx': fold_i_idx,
            'mean': mean_performance,
            'std': std_performance,
            'features': selected_features_i.copy()
        })
        
        if verbose:
            print(f"  Fold {fold_i_idx + 1} mean {metric}: {mean_performance:.4f} (std: {std_performance:.4f})")
    
    # Select best fold based on task type
    if task_type == 'classification':
        # Higher is better
        best_fold_info = max(all_fold_performances, key=lambda x: x['mean'])
        if verbose:
            print(f"\nBest fold: Fold {best_fold_info['fold_idx'] + 1} with mean {metric}: {best_fold_info['mean']:.4f} (std: {best_fold_info['std']:.4f})")
    else:  # regression
        # Lower is better
        best_fold_info = min(all_fold_performances, key=lambda x: x['mean'])
        if verbose:
            print(f"\nBest fold: Fold {best_fold_info['fold_idx'] + 1} with mean {metric}: {best_fold_info['mean']:.4f} (std: {best_fold_info['std']:.4f})")
    
    return best_fold_info['features'], best_fold_info['fold_idx'], all_fold_performances

def inference(final_features, nFold, feature_df, score_df, shuffle_flag, random_seed, specialist_features, balance, model_name, model, metric, task_type, probability):    
    kf5 = KFold(n_splits = nFold, shuffle=shuffle_flag, random_state=random_seed)
    valPerformanceByFold = []
    actual_score = []
    predicted_score = []
    subjects = []
    predicted_probs = []
    
    # Generating subject IDs (sub1, sub2, etc.)
    subject_ids = ['sub' + str(i+1) for i in range(len(feature_df))]
    
    for infer_fold in kf5.split(feature_df):
        train_val_df = feature_df.iloc[infer_fold[0]]
        test_df = feature_df.iloc[infer_fold[1]]
        
        # Getting the subject IDs for the test set
        test_subjects = [subject_ids[i] for i in infer_fold[1]]
        
        if len(specialist_features) != 0:
            train_specialist = specialist_features.iloc[infer_fold[0]]
            test_specialist = specialist_features.iloc[infer_fold[1]]
        
        # Training
        df_s = score_df.iloc[infer_fold[0]]
        df_f = train_val_df[final_features]
        
        # Validation
        df_val_s = score_df.iloc[infer_fold[1]]
        df_val_f = test_df[final_features]
        
        if len(specialist_features) != 0:
            df_f = pd.concat([train_specialist, df_f], axis=1)
            df_val_f = pd.concat([test_specialist, df_val_f], axis=1)
        
        if model_name == 'consensus':
            predictions = []
            probabilities = []
            
            for one_model in model:
                model_ = model[one_model]
                
                # Fit data to a model with the selected features and predict
                model_.fit(df_f, df_s.values.ravel())
                y_pred = model_.predict(df_val_f)
                predictions.append(y_pred)
                
                if probability == True:
                    y_pred_proba = model_.predict_proba(df_val_f)
                    prob_class_1 = y_pred_proba[:, 1] # probability of class 1
                    probabilities.append(prob_class_1)
                    
            if probability == True:
                # Averaging the probabilities from the 3 models
                consensus_prob = [(a+b+c)/3 for a,b,c in zip(probabilities[0], probabilities[1], probabilities[2])]
                predicted_probs.append(consensus_prob)
            
            if task_type == 'regression':
                consensus_pred = [(a+b+c)/3 for a,b,c in zip(predictions[0], predictions[1], predictions[2])]
            elif task_type == 'classification':
                def majority_vote(a,b,c):
                    return 1 if a+b+c>1 else 0
                consensus_pred = [majority_vote(a,b,c) for a,b,c in zip(predictions[0], predictions[1], predictions[2])]
                   
            # Calculating the performance for the validation set
            performance = loss_estimation(metric, df_val_s, consensus_pred)
                
            # Saving the actual, predicted scores, and subject IDs
            actual_score.append(df_val_s.values.ravel().tolist())
            predicted_score.append(consensus_pred)
            subjects.append(test_subjects)
            
            valPerformanceByFold.append(performance)
        else:
            # Fitting data to a model with the selected features and predict
            model.fit(df_f, df_s.values.ravel())
            y_pred = model.predict(df_val_f)
            
            if probability == True and task_type == 'classification':
                y_pred_proba = model.predict_proba(df_val_f)
                prob_class_1 = y_pred_proba[:, 1]  # probability of class 1
                predicted_probs.append(prob_class_1)
                
            # Calculating the performance for the validation set
            valPerformanceByFold.append(loss_estimation(metric, df_val_s, y_pred))
            actual_score.append(df_val_s.values.ravel().tolist())
            predicted_score.append(y_pred.tolist())
            subjects.append(test_subjects)
        
    # Flattening the lists of actual scores, predicted scores, and subject IDs
    actual = [item for sublist in actual_score for item in sublist]
    predicted = [round(item, 2) for sublist in predicted_score for item in sublist]
    subjects = [item for sublist in subjects for item in sublist]
    
    if probability == True:
        predicted_probs = [round(item, 2) for sublist in predicted_probs for item in sublist]
        return subjects, actual, [predicted]+[predicted_probs], valPerformanceByFold
    else:
        return subjects, actual, predicted, valPerformanceByFold

def inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability):
    valPerformanceByFold = []
    actual_score = []
    predicted_score = []
    subjects = []
    predicted_probs = []
    
    # Preparing the validation features
    if len(specialist_features) != 0:
        df_val_f = pd.concat([specialist_features, feature_df[final_features]], axis=1)
    else:
        df_val_f = feature_df[final_features]
 
    specialist_features = pd.DataFrame(specialist_features)
    specialist_feature_names = list(specialist_features.columns)
    all_column_names = specialist_feature_names + final_features

    inference_data_df2 = inference_data_df[all_column_names].copy()

    # Extracting subject IDs (e.g., sub1, sub2, sub3,...)
    inference_subjects = ["sub" + str(i + 1) for i in range(len(inference_data_df))]
    
    if model_name == 'consensus':
        predictions = []
        for one_model in model:
            model_ = model[one_model]
            
            # Fit data to a model with the selected features and predict
            model_.fit(df_val_f, score_df.values.ravel())
            y_pred = model_.predict(inference_data_df2)
            predictions.append(y_pred)
            
            if probability == True:
                y_pred_proba = model_.predict_proba(inference_data_df2)
                prob_class_1 = y_pred_proba[:, 1] # probability of class 1
                probabilities.append(prob_class_1)
        
        if probability == True:
            # Averaging the probabilities from the 3 models
            consensus_prob = [(a+b+c)/3 for a,b,c in zip(probabilities[0], probabilities[1], probabilities[2])]
            predicted_probs.append(consensus_prob)
        
        if task_type == 'regression':
            consensus_pred = [(a+b+c)/3 for a,b,c in zip(predictions[0],predictions[1],predictions[2])]
        elif task_type == 'classification':
            def majority_vote(a,b,c):
                return 1 if a+b+c>1 else 0
            consensus_pred = [majority_vote(a,b,c) for a,b,c in zip(predictions[0],predictions[1],predictions[2])]
        
        # Calculate the performance for the validation set
        performance = loss_estimation(metric, df_val_s, consensus_pred)
            
        # Save the actual and prediction scores
        actual_score.append(df_val_s.values.ravel().tolist())
        predicted_score.append(consensus_pred)
        subjects.append(inference_subjects)
        
        valPerformanceByFold.append(performance)
    else:
        # Fit data to a model with the selected features and predict
        model.fit(df_val_f, score_df.values.ravel())
        y_pred = model.predict(inference_data_df2)
        
        if probability == True and task_type == 'classification':
            y_pred_proba = model.predict_proba(inference_data_df2)
            prob_class_1 = y_pred_proba[:, 1]  # probability of class 1
            predicted_probs.append(prob_class_1)
            
        # Calculate the mean absolute error for the validation set
        valPerformanceByFold.append(loss_estimation(metric, inference_score_df, y_pred))
        
        actual_score.append(inference_score_df.values.ravel().tolist())
        predicted_score.append(y_pred.tolist())
        subjects.append(inference_subjects)
        
    # Flatten the lists of actual scores, predicted scores, and subject IDs
    actual = [item for sublist in actual_score for item in sublist]
    predicted = [round(item, 2) for sublist in predicted_score for item in sublist]
    subjects = [item for sublist in subjects for item in sublist]

    if probability == True:
        predicted_probs = [round(item, 2) for sublist in predicted_probs for item in sublist]
        return subjects, actual, [predicted]+[predicted_probs], valPerformanceByFold
    else:
        return subjects, actual, predicted, valPerformanceByFold

        
def loss_estimation(metric, true_values, predicted_values):
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
        
    if metric == 'MAE':
        mae = mean_absolute_error(true_values, predicted_values)
        return mae
    elif metric == 'MAPE':
        true_values_no_zero = np.where(true_values == 0, 1e-10, true_values)
        mape = np.mean(np.abs((true_values - predicted_values) / true_values_no_zero)) * 100
        return mape
    elif metric == 'Accuracy':
        accuracy = accuracy_score(true_values, predicted_values)
        return accuracy
    elif metric == 'binaryROC':
        cm  = confusion_matrix(true_values, predicted_values)
        tn, fp, fn, tp = cm.ravel()
        epsilon = 1e-7
        sensitivity = round(tp / (tp + fn + epsilon), 2)
        specificity = round(tn / (tn + fp + epsilon), 2)
        binaryROC = ((1-sensitivity) ** 2) + ((1-specificity) ** 2)
        return binaryROC
    elif metric == 'F1-score':
        cm  = confusion_matrix(true_values, predicted_values)
        tn, fp, fn, tp = cm.ravel()
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        f1_score = round(f1_score, 2)
        return f1_score
    else:
        raise ValueError("Unknown metric")
    
def balanceData(data, target):
    # Concatenate them for easier manipulation
    df = pd.concat([target, data], axis=1)
    df.columns = ['groundTruth'] + list(data.columns)

    # Determine which class is the majority and which is the minority
    class_counts = df['groundTruth'].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    # Separate majority and minority classes
    df_majority = df[df['groundTruth'] == majority_class]
    df_minority = df[df['groundTruth'] == minority_class]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=123) 

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Separate features and target from the balanced dataframe
    feature_balanced = df_upsampled.drop('groundTruth', axis=1)
    class_balanced = df_upsampled['groundTruth']
    
    return feature_balanced, class_balanced
