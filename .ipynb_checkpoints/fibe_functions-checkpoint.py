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
    probability: if True, probability values (for the class 1) that leads to binary classifications is returned. This option only works when the 'task_type=regression.'
    balance: In a binary classification task, if the data is imbalanced in terms of classes, 'balance=True' uses resampling to balance the data.
    model_name: For the 'regression' task, choose from 'linearSVR', 'gaussianSVR', 'RegressionForest', 'AdaBoostDT', 'AdaBoostSVR', 
            and 'consensus' (consensus using 'linearSVR', 'gaussianSVR', and 'RegressionForest'). The default is 'linearSVR'. For 'classification' task, 
            to choose from 'linearSVC', 'gaussianSVC', 'RandomForest', 'AdaBoostDT', 'AdaBoostSVC', and 'consensus' (consensus using 'linerSVC', 
            'gaussianSVC', and 'RandomForest'). The default is 'linearSVC'.
    metric: For the 'regression' task, choose from 'MAE' and 'MAPE'. The default is 'MAE.' For the 'classification' task, choose from 'Accuracy', 
            'F1-score,' and 'binaryROC'. The default is 'Accuracy'.
    voting_strictness: either 'strict' that chooses those features that is selected at least 3 times in 5-fold cross-validation, or 
            'loose' that chooses those features that is selected at least 2 times in 5-fold cross-validation, or 'both' that chooses both strict and loose options. The default is 'strict'.
            For any random number of folds, N, the 'strict' threshold should be 0.6 X N and the 'loose' threshold should be 0.4 X N.
    nFold: Number of folds in cross-validation. Preferred and default is '5'.
    maxIter: is the maximum number of iterations that the algorithm goes back and forth in forward inclusion and backward elimination in each fold. The default is '3'.
    tolerance: is the percentage of deviation in the error/accuracy threshold allowed. The default is '0.05', i.e., 5%.
    maxFeatures: is the fractional number that indicate the number of best features to be selected of the total features. Default is 0.25, i.e., 25% of the total number of features.
    save_intermediate: if True, saves intermediate results to the specified directory. Default is False.
    output_dir: directory where intermediate results are saved if save_intermediate is True.
    inference_data_df: data for optional second inference cohort for prediction using the selected subset of features.
    inference_score_df: scores for optional second inference cohort for prediction using the selected subset of features.
    verbose: generates text for intermediate loss and selected feature list during iteration. The default is 'True'.

    The outputs are:
    subjectList: is the list for subjects used in inference. Each subject/patient is assigned a name as 'subXX' and according to this list, other outputs are organized in the subsequent generated lists.
    selectedFeatures: is the list of features if 'columns_names' was not 'None'. Otherwise column indexes of the selected features. For 'voting_strictness' of 'both', 
            'selectedFeatures' contains two sets of output as [[selected features for 'strict'], [selected feature for 'loose']].
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
            model = AdaBoostRegressor(estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'AdaBoostSVR':
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
            model = AdaBoostClassifier(estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'AdaBoostSVC':
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
        voting_strictness = 'strict'  # Default 
        
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
    elif voting_strictness == 'both':
        vote = 0
    else:
        raise ValueError("Unknown voting strictness. Must be either 'strict' or 'loose.'")
        
    if tolerance == None:
        tolerance = 0.05  # Default 
    elif tolerance == 1:
        raise ValueError("tolerance cannot be 1.")
        
    if maxFeatures == None:
        maxFeatures = round(0.25*feature_df.shape[1])  # Default 
    elif maxFeatures > feature_df.shape[1]:
        raise ValueError("maxFeatures cannot be greater than the umber of features available.")
    
    if save_intermediate and output_dir is None:
        raise ValueError("Directory for saving intermediate results is not provided.")


    flag_union = 0
    
    # training a model
    selectedFeatures = train(maxIter, nFold, feature_df, score_df, specialist_features, task_type, balance, model_name, model, metric, tolerance, maxFeatures, save_intermediate, output_dir, verbose)
    
    # inference
    if probability == True:
        model = model_infer
        
    if vote == round(0.6 * nFold) or vote == round(0.4 * nFold):
        final_features = [element for element, count in selectedFeatures.items() if count >= vote]
        if len(final_features) >= 2:
            subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
        else:
            flag_union = 1
    elif vote == 0:
        f_features = []
        ac_score = []
        pr_score = []
        val_per = []
        sub_list = []
        
        # for strict choice
        final_features = [element for element, count in selectedFeatures.items() if count >= round(0.6 * nFold)]
        if len(final_features) >= 2:
            subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
            f_features = f_features + [final_features]
            ac_score = ac_score + [actual_score]
            pr_score = pr_score + [predicted_score]
            val_per = val_per + [validationPerformance]
            sub_list = sub_list + [subjectList]
        else:
            flag_union = 1
        
        # for loose choice
        final_features = [element for element, count in selectedFeatures.items() if count >= round(0.4 * nFold)]
        if len(final_features) >= 2:
            subjectList, actual_score, predicted_score, validationPerformance = inference(final_features, nFold, feature_df, score_df, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + final_features
            f_features = f_features + [final_features]
            ac_score = ac_score + [actual_score]
            pr_score = pr_score + [predicted_score]
            val_per = val_per + [validationPerformance]
            sub_list = sub_list + [subjectList]

            final_features = f_features
            actual_score = ac_score
            predicted_score = pr_score
            validationPerformance = val_per
            subjectList = sub_list
        else:
            flag_union = 1
            
    if flag_union == 1:
        union_list = list(selectedFeatures.keys())
        subjectList, actual_score, predicted_score, validationPerformance = inference(union_list, nFold, feature_df, score_df, specialist_features, balance, model_name, model, metric, task_type, probability)   # Added task_type
        if len(specialist_features) != 0:
            final_features = list(specialist_features.columns) + union_list
        else:
            final_features = union_list
        
    # inference on additional data
    if inference_data_df is not None and inference_score_df is not None:
        flag_union = 0
        if vote == round(0.6 * nFold) or vote == round(0.4 * nFold):
            final_features = [element for element, count in selectedFeatures.items() if count >= vote]
            if len(final_features) >= 2:
                subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
                if len(specialist_features) != 0:
                    final_features = list(specialist_features.columns) + final_features
            else:
                flag_union = 1
                
        elif vote == 0:
            ac_score_add = []
            pr_score_add = []
            val_per_add = []
            sub_list_add = []
            
            # for strict choice
            final_features = [element for element, count in selectedFeatures.items() if count >= round(0.6 * nFold)]
            if len(final_features) >= 2:
                subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
                ac_score_add = ac_score_add + [actual_score_add]
                pr_score_add = pr_score_add + [predicted_score_add]
                val_per_add = val_per_add + [validationPerformance_add]
                sub_list_add = sub_list_add + [subjectList_add]
            else:
                flag_union = 1
            
            # for loose choice
            final_features = [element for element, count in selectedFeatures.items() if count >= round(0.4 * nFold)]
            if len(final_features) >= 2:
                subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(final_features, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)    # Added task_type
                ac_score_add = ac_score_add + [actual_score_add]
                pr_score_add = pr_score_add + [predicted_score_add]
                val_per_add = val_per_add + [validationPerformance_add]
                sub_list_add = sub_list_add + [subjectList_add]

                actual_score_add = ac_score_add
                predicted_score_add = pr_score_add
                validationPerformance_add = val_per_add
                subjectList_add = sub_list_add
            else:
                flag_union = 1

        actual_score = actual_score + [actual_score_add]  
        predicted_score = predicted_score + [predicted_score_add]
        validationPerformance = validationPerformance + [validationPerformance_add]
        
        if flag_union == 1:
            union_list = list(selectedFeatures.keys())
            subjectList_add, actual_score_add, predicted_score_add, validationPerformance_add = inference_additional(union_list, feature_df, score_df, specialist_features, inference_data_df, inference_score_df, model_name, model, metric, task_type, probability)
            if len(specialist_features) != 0:
                final_features = list(specialist_features.columns) + union_list
            else:
                final_features = union_list
                
        actual_score = actual_score + [actual_score_add]  
        predicted_score = predicted_score + [predicted_score_add]
        validationPerformance = validationPerformance + [validationPerformance_add]
        subjectList = subjectList + [subjectList_add]

    return final_features, subjectList, actual_score, predicted_score, validationPerformance
        
        
def train(maxIter, nFold, feature_df, score_df, specialist_features, task_type, balance, model_name, model, metric, tolerance, maxFeatures, save_intermediate, output_dir, verbose=False):
    max_iter = maxIter
    kf5 = KFold(n_splits = nFold, shuffle=True, random_state=42)
    
    frequency_of_features_selected_all_fold = []

    oF = 0
    for outer_fold in kf5.split(feature_df):
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
                    if np.min(temp_error) < lowest_error*(1+tolerance) and len(selected_features) < maxFeatures:
                        selected_features.append(feature_df.columns[np.argmin(temp_error)])
                        
                        if np.min(temp_error) < lowest_error:
                            lowest_error = np.min(temp_error)
                            best_features = copy.deepcopy(selected_features)
                             
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Error ({metric}): {lowest_error:.4f} | Current Error ({metric}): {np.min(temp_error):.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Error ({metric}): {lowest_error:.4f} | Current Error ({metric}): {np.min(temp_error):.4f} | Selected Features: {selected_features}", flush=True)
                    else:
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + best_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] No additional feature improves performance beyond tolerance | Best Features: {all_feat} | Starting BE..", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] No additional feature improves performance beyond tolerance | Best Features: {best_features} | Starting BE..", flush=True)
                        flag_FI = 1
                        break
                else:
                    if np.max(temp_error) > highest_accuracy*(1-tolerance) and len(selected_features) < maxFeatures:
                        selected_features.append(feature_df.columns[np.argmax(temp_error)])
                        
                        if np.max(temp_error) > highest_accuracy:
                            highest_accuracy = np.max(temp_error)
                            best_features = copy.deepcopy(selected_features)
                            
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Prediction ({metric}): {highest_accuracy:.4f} | Currenct Prediction ({metric}): {np.max(temp_error):.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {oF} | Iter: {i+1} | FI | Traversal: {q+1}] - Traversal over all features finished | Best Prediction ({metric}): {highest_accuracy:.4f} | Currenct Prediction ({metric}): {np.max(temp_error):.4f} | Selected Features: {selected_features}", flush=True)
                    
                    else: 
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
                    print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}]", flush=True)
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
                            print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] No removal of additional feature improves performance. | Selected Features: {selected_features}", flush=True)
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
                            print(f"[Fold: {oF} | Iter: {i+1} | BE | Traversal: {q+1}] No removal of additional feature improves performance. | Selected Features: {selected_features}", flush=True)
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
        frequency_of_features_selected_all_fold = frequency_of_features_selected_all_fold + selected_features
        
    feature_counts = Counter(frequency_of_features_selected_all_fold)
    return feature_counts

def inference(final_features, nFold, feature_df, score_df, specialist_features, balance, model_name, model, metric, task_type, probability):    
    kf5 = KFold(n_splits = nFold, shuffle=True, random_state=42)
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
