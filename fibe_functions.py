import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import copy
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

def fibe(feature_df, score_df, columns_names=None, task_type=None, model_name=None, metric=None, voting_strictness=None, nFold=None, maxIter=None, verbose=True):
    
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
            raise ValueError("Unknown model name")
    else:
        if model_name == None or model_name == 'linearSVC':
            model = SVC(kernel = 'linear', C=1.0)  # Default
        elif model_name == 'gaussianSVR':
            model = SVC(kernel = 'rbf', C=1.0, gamma='scale')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators = 100, random_state=42, max_depth=5)
        elif model_name == 'AdaBoostDT':
            model = AdaBoostClassifier(estimator=None, n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'AdaBoostSVR':
            model = AdaBoostClassifier(estimator=SVC(), algorithm = 'SAMME', n_estimators=50, learning_rate=1.0, random_state=42)
        elif model_name == 'consensus':
            model = { 
            "Random Forest" : RandomForestClassifier(n_estimators = 100, random_state=42, max_depth=5),
            "Gaussian SVR" : SVC(kernel = 'rbf', C=1.0, gamma='scale'), 
            "Linear SVR" : SVC(kernel = 'linear', C=1.0)
            }
        else:
            raise ValueError("Unknown model name")
        
    if task_type == 'regression' and metric == None:
        metric = 'MAE' # Default 
    elif task_type == 'classification' and metric == None:
        metric = 'Accuracy'   # Default 
        
    if voting_strictness == None:
        voting_strictness = 'strict'  # Default 
        
    if nFold == None:
        nFold = 5   # Default 
        
    if maxIter == None:
        maxIter = 3  # Default 
        
    if voting_strictness == 'strict':
        vote = 3
    elif voting_strictness == 'loose':
        vote = 2
    else:
        raise ValueError("Unknown voting strictness. Must be either 'strict' or 'loose.'")
        
    # training a model
    selectedFeatures = train(maxIter, nFold, feature_df, score_df, task_type, model_name, model, metric, verbose)
    final_features = [element for element, count in selectedFeatures.items() if count >= vote]
    
    # inference
    validationPerformance = inference(final_features, nFold, feature_df, score_df, model_name, model, metric)
    
    return final_features, validationPerformance
        
        
def train(maxIter, nFold, feature_df, score_df, task_type, model_name, model, metric, verbose=False):
    max_iter = maxIter
    kf5 = KFold(n_splits = nFold, shuffle = False)
    
    frequency_of_features_selected_all_fold = []

    oF = 0
    for outer_fold in kf5.split(feature_df):
        train_val_df = feature_df.iloc[outer_fold[0]]
        test_df = feature_df.iloc[outer_fold[1]]
        
        train_val_score_df = score_df.iloc[outer_fold[0]]
        test_score_df = score_df.iloc[outer_fold[1]]
        
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
                    
                    # training
                    df_score = train_val_score_df.iloc[inner_fold[0]]
                    df_feature = training[temp_features]
                    
                    # validation
                    df_val_score = train_val_score_df.iloc[inner_fold[1]]
                    df_val_feature = validation[temp_features]
                    
                    if model_name == 'consensus':
                        for one_model in model:
                            model_ = model[one_model]
                            
                            # Train a Regressor with the selected features
                            model_.fit(df_feature, df_score.values.ravel())
                        
                            # Predict on the validation set
                            y_pred = model_.predict(df_val_feature)
                                
                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                    else:
                        # Train a Regressor with the selected features
                        model.fit(df_feature, df_score.values.ravel())
                            
                        # Predict on the validation set
                        y_pred = model.predict(df_val_feature)
                            
                        # Calculate the mean absolute error for the validation set
                        inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                
                if task_type == 'regression':
                    if np.mean(inner_error) < lowest_error:
                        lowest_error = np.mean(inner_error)
                        selected_features.append(feature)
                        if verbose:
                            print(f"====== Features: {selected_features}, {metric}: {lowest_error:.4f}")
                else:
                    if np.mean(inner_error) > highest_accuracy:
                        highest_accuracy = np.mean(inner_error)
                        selected_features.append(feature)
                        if verbose:
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
                    
                    # training
                    df_score = train_val_score_df.iloc[inner_fold[0]]
                    df_feature = training[temp_features]
                    
                    # validation
                    df_val_score = train_val_score_df.iloc[inner_fold[1]]
                    df_val_feature = validation[temp_features]
                    
                    if model_name == 'consensus':
                        for one_model in model:
                            model_ = model[one_model]
                            
                            # Train a Regressor with the selected features
                            model_.fit(df_feature, df_score.values.ravel())
                        
                            # Predict on the validation set
                            y_pred = model_.predict(df_val_feature)
                                
                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                    else:
                        # Train a Regressor with the selected features
                        model.fit(df_feature, df_score.values.ravel())
                            
                        # Predict on the validation set
                        y_pred = model.predict(df_val_feature)
                            
                        # Calculate the mean absolute error for the validation set
                        inner_error.append(loss_estimation(metric, df_val_score, y_pred))
                
                if task_type == 'regression':
                    if np.mean(inner_error) < lowest_error:
                        lowest_error = np.mean(inner_error)
                        selected_features.remove(feature)
                        if verbose:
                            print(f"====== Features: {selected_features}, {metric}: {lowest_error:.4f}")
                else:
                    if np.mean(inner_error) > highest_accuracy:
                        highest_accuracy = np.mean(inner_error)
                        selected_features.remove(feature)
                        if verbose:
                            print(f"====== Features: {selected_features}, {metric}: {highest_accuracy:.4f}")
        
        # saving features selected across all outer folds
        frequency_of_features_selected_all_fold = frequency_of_features_selected_all_fold + selected_features
        
    feature_counts = Counter(frequency_of_features_selected_all_fold)
    return feature_counts


def inference(final_features, nFold, feature_df, score_df, model_name, model, metric):
    kf5 = KFold(n_splits = nFold, shuffle = False)
    valPerformanceByFold = []
    
    for infer_fold in kf5.split(feature_df):
        train_val_df = feature_df.iloc[infer_fold[0]]
        test_df = feature_df.iloc[infer_fold[1]]
        
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
                
                # Fit data to a model with the selected features
                model_.fit(df_f, df_s.values.ravel())
            
                # Predict on the validation set
                y_pred = model_.predict(df_val_f)
                    
                # Calculate the performance for the validation set
                accumulated_performance.append(loss_estimation(metric, df_val_s, y_pred))
            
            valPerformanceByFold.append(np.mean(accumulated_performance))
        else:
            # Fit data to a model with the selected features
            model.fit(df_f, df_s.values.ravel())
                
            # Predict on the validation set
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
    else:
        raise ValueError("Unknown metric")