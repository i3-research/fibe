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
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
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

def fibe(feature_df, score_df, fixed_features=None, columns_names=None, task_type=None, model_name=None, metric=None, voting_strictness=None, nFold=None, maxIter=None, verbose=True):
    
    '''
    feature_df: is the 2D feature matrix (supports DataFrame, Numpy Array, and List) with columns representing different features.
    score_df: is the 1D score vector as a column (supports DataFrame, Numpy Array, and List).
    fixed_features: Predefined features that must stay in the feature set and the FIBE algorithm does not add or remove those. 
            Must be either a List of names to select from 'feature_df', or DataFrame of features added separately to 'feature_df.'
    columns_names: contain the names of the features. The algorithm returns the names of the selected features from this list. 
            If not available, then the algorithm returns the column indexes of selected features. 
    task_type: either 'regression' or 'classification.' Default is 'regression.'
    model_name: For 'regression' task, to choose from 'linerSVR', 'gaussianSVR', 'RegressionForest', 'AdaBoostDT', 'AdaBoostSVR', 
            and 'consensus' (consensus using 'linerSVR', 'gaussianSVR', and 'RegressionForest'). Default is 'linerSVR'. For 'classification' task, 
            to choose from 'linerSVC', 'gaussianSVC', 'RandomForest', 'AdaBoostDT', 'AdaBoostSVC', and 'consensus' (consensus using 'linerSVC', 
            'gaussianSVC', and 'RandomForest'). Default is 'linerSVC'.
    metric: For 'regression' task, to choose from 'MAE' and 'MAPE'. Default is 'MAE.' For 'classification' task, to choose from 'Accuracy', 
            'F1-score,' and 'binaryROC'. Default is 'Accuracy'.
    voting_strictness: either 'strict' that chooses those features that is selected at least 3 times in 5-fold cross-validation, or 
            'loose' that chooses those features that is selected at least 2 times in 5-fold cross-validation. Default is 'strict'.
            For any random number of folds, N, 'strict' threshold should be 0.6 X N and 'loose' threshold should be 0.4 X N.
    nFold: Number of folds in cross-validation. Preferred and default is '5'.
    maxIter: is the maximum number of iteration that the algorithm goes back and forth in forward inclusion and backward elimination in each fold. Default is '3'.
    verbose: generates text for intermediate loss and selected feature list during iteration. Default is 'True'.

    The outputs are:
    selectedFeatures: is the list of features if 'columns_names' was not 'None'. Otherwise column indexes of the selected features.
    validationPerformance: is a list containing validation performance in terms of chosen 'metric' for 'nFold' folds.
    '''
    
    
    # checking and converting to DataFrame if the features are in numpy array format
    if type(feature_df) is np.ndarray:
        if columns_names != None:
            # checking if the length of feature names are equal to the feature column numbers
            if feature_df.shape[1] != len(columns_names):
                raise ValueError("Number of columns in the Feature is not equal to the number of column names provided")
            else:
                feature_df = pd.DataFrame(data=feature_df, columns=columns_names)
        else:
            feature_df = pd.DataFrame(data=feature_df)
            
    # checking and converting to DataFrame if the features are in list format
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
    
    # checking and converting to DataFrame if the target is in list format
    elif isinstance(score_df, list):
        score_df = pd.DataFrame(data=score_df)
    
    # checking if the observation/subject numbers are equal in features and in targets
    if feature_df.shape[0] != score_df.shape[0]:
        raise ValueError("Number of rows in the Feature is not equal to the number of rows in Score")
    
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
    
    # Assigning a model if None, or, assigning models based on task type
    if task_type == 'regression':
        if model_name == None or model_name == 'linearSVR':
            model = SVR(kernel = 'linear', C=1.0, epsilon=0.2)  # Default
        elif model_name == 'gaussianSVR':
            model = SVR(kernel = 'rbf', C=1.0, gamma='scale')
        elif model_name == 'RegressionForest':
            model = RandomForestRegressor(n_estimators = 100, random_state=42, max_depth=5)
        elif model_name == 'AdaBoostDT':
            model = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'AdaBoostSVR':
            model = AdaBoostRegressor(base_estimator=SVR(), n_estimators=50, learning_rate=1.0, random_state=42)
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
        elif model_name == 'gaussianSVC':
            model = SVC(kernel = 'rbf', C=1.0, gamma='scale')
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
        vote = 3
    elif voting_strictness == 'loose':
        vote = 2
    else:
        raise ValueError("Unknown voting strictness. Must be either 'strict' or 'loose.'")
        
    # training a model
    selectedFeatures = train(maxIter, nFold, feature_df, score_df, specialist_features, task_type, model_name, model, metric, verbose)
    final_features = [element for element, count in selectedFeatures.items() if count >= vote]
    if len(specialist_features) != 0:
        final_features = list(specialist_features.columns) + final_features
    
    # inference
    validationPerformance = inference(final_features, nFold, feature_df, score_df, specialist_features, model_name, model, metric)
    
    return final_features, validationPerformance
        
        
def train(maxIter, nFold, feature_df, score_df, specialist_features, task_type, model_name, model, metric, verbose=False):
    max_iter = maxIter
    kf5 = KFold(n_splits = nFold, shuffle = False)
    
    frequency_of_features_selected_all_fold = []

    oF = 0
    for outer_fold in kf5.split(feature_df):
        train_val_df = feature_df.iloc[outer_fold[0]]
        train_val_score_df = score_df.iloc[outer_fold[0]]
        if len(specialist_features) != 0:
            train_specialist = specialist_features.iloc[outer_fold[0]]
        
        oF += 1
        
        if verbose:
            print(f'Outer Fold: {oF}')
            
        selected_features = []
        
        if task_type == 'regression':
            lowest_error = float('inf')
        else:
            highest_accuracy = float('-inf')
        
        for i in range(max_iter):
            if verbose:
                print(f'==== Iteration: {i+1}')
            
            for feature in feature_df.columns:
                if feature in selected_features:
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
                    
                    if model_name == 'consensus':
                        for one_model in model:
                            model_ = model[one_model]
                            
                            # Train a Regressor with the selected features and predict
                            if len(specialist_features) != 0:
                                model_.fit(pd.concat([train_spec, df_feature], axis=1), df_score.values.ravel())
                                y_pred = model_.predict(pd.concat([valid_spec, df_val_feature], axis=1))
                            else:
                                model_.fit(df_feature, df_score.values.ravel())
                                y_pred = model_.predict(df_val_feature)
                                
                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                    else:
                        # Train a Regressor with the selected features and predict
                        if len(specialist_features) != 0:
                            model.fit(pd.concat([train_spec, df_feature], axis=1), df_score.values.ravel())
                            y_pred = model.predict(pd.concat([valid_spec, df_val_feature], axis=1))
                        else:
                            model.fit(df_feature, df_score.values.ravel())
                            y_pred = model.predict(df_val_feature)
                            
                        # Calculate the mean absolute error for the validation set
                        inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                
                if task_type == 'regression':
                    if np.mean(inner_error) < lowest_error:
                        lowest_error = np.mean(inner_error)
                        selected_features.append(feature)
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"====== Features: {all_feat}, {metric}: {lowest_error:.4f}")
                            else:
                                print(f"====== Features: {selected_features}, {metric}: {lowest_error:.4f}")
                else:
                    if np.mean(inner_error) > highest_accuracy:
                        highest_accuracy = np.mean(inner_error)
                        selected_features.append(feature)
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"====== Features: {all_feat}, {metric}: {highest_accuracy:.4f}")
                            else:
                                print(f"====== Features: {selected_features}, {metric}: {highest_accuracy:.4f}")
            
            # Backward Elimination
            if verbose:
                print(f'===== FI finished, BE begins ======')
            selected_features_ = copy.deepcopy(selected_features)
            
            for feature in selected_features_:
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
                    
                    if model_name == 'consensus':
                        for one_model in model:
                            model_ = model[one_model]
                            
                            # Train a Regressor with the selected features and predict
                            if len(specialist_features) != 0:
                                model_.fit(pd.concat([train_spec, df_feature], axis=1), df_score.values.ravel())
                                y_pred = model_.predict(pd.concat([valid_spec, df_val_feature], axis=1))
                            else:
                                model_.fit(df_feature, df_score.values.ravel())
                                y_pred = model_.predict(df_val_feature)
                                                    
                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                    else:
                        # Train a Regressor with the selected features and predict
                        if len(specialist_features) != 0:
                            model.fit(pd.concat([train_spec, df_feature], axis=1), df_score.values.ravel())
                            y_pred = model.predict(pd.concat([valid_spec, df_val_feature], axis=1))
                        else:
                            model.fit(df_feature, df_score.values.ravel())
                            y_pred = model.predict(df_val_feature)
                            
                        # Calculate the mean absolute error for the validation set
                        inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                
                if task_type == 'regression':
                    if np.mean(inner_error) < lowest_error:
                        lowest_error = np.mean(inner_error)
                        selected_features.remove(feature)
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"====== Features: {all_feat}, {metric}: {lowest_error:.4f}")
                            else:
                                print(f"====== Features: {selected_features}, {metric}: {lowest_error:.4f}")
                else:
                    if np.mean(inner_error) > highest_accuracy:
                        highest_accuracy = np.mean(inner_error)
                        selected_features.remove(feature)
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"====== Features: {all_feat}, {metric}: {highest_accuracy:.4f}")
                            else:
                                print(f"====== Features: {selected_features}, {metric}: {highest_accuracy:.4f}")
        
        # saving features selected across all outer folds
        frequency_of_features_selected_all_fold = frequency_of_features_selected_all_fold + selected_features
        
    feature_counts = Counter(frequency_of_features_selected_all_fold)
    return feature_counts


def inference(final_features, nFold, feature_df, score_df, specialist_features, model_name, model, metric):
    kf5 = KFold(n_splits = nFold, shuffle = False)
    valPerformanceByFold = []
    
    for infer_fold in kf5.split(feature_df):
        train_val_df = feature_df.iloc[infer_fold[0]]
        test_df = feature_df.iloc[infer_fold[1]]
        
        if len(specialist_features) != 0:
            train_specialist = specialist_features.iloc[infer_fold[0]]
            test_specialist = specialist_features.iloc[infer_fold[1]]
        
        # training
        df_s = score_df.iloc[infer_fold[0]]
        df_f = train_val_df[final_features]
        
        # validation
        df_val_s = score_df.iloc[infer_fold[1]]
        df_val_f = test_df[final_features]
        
        if model_name == 'consensus':
            accumulated_performance = []
            for one_model in model:
                model_ = model[one_model]
                
                # Fit data to a model with the selected features and predict
                if len(specialist_features) != 0:
                    model_.fit(pd.concat([train_specialist, df_f], axis=1), df_s.values.ravel())
                    y_pred = model_.predict(pd.concat([test_specialist, df_val_f], axis=1))
                else:
                    model_.fit(df_f, df_s.values.ravel())
                    y_pred = model_.predict(df_val_f, axis=1)
                    
                # Calculate the performance for the validation set
                accumulated_performance.append(loss_estimation(metric, df_val_s, y_pred))
            
            valPerformanceByFold.append(np.mean(accumulated_performance))
        else:
            # Fit data to a model with the selected features and predict
            if len(specialist_features) != 0:
                model.fit(pd.concat([train_specialist, df_f], axis=1), df_s.values.ravel())
                y_pred = model.predict(pd.concat([test_specialist, df_val_f], axis=1))
            else:
                model.fit(df_f, df_s.values.ravel())
                y_pred = model.predict(df_val_f)
                
            # Calculate the mean absolute error for the validation set
            valPerformanceByFold.append(loss_estimation(metric, df_val_s, y_pred))
            
    return valPerformanceByFold
        
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
        tn, fp, fn, tp  = confusion_matrix(true_values, predicted_values)
        sensitivity = round(tp / (tp + fn),2)
        specificity = round(tn / (tn + fp),2)
        binaryROC = ((1-sensitivity) ** 2) + ((1-specificity) ** 2)
        return binaryROC
    elif metric == 'F1-score':
        tn, fp, fn, tp  = confusion_matrix(true_values, predicted_values)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = round(f1_score, 2)
        return f1_score
    else:
        raise ValueError("Unknown metric")