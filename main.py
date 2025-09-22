# Import necessary libraries
import pandas as pd
from sklearn import datasets
import numpy as np
from fibe_functions import fibe

#======================================
#      Regression Example
#======================================

# Load Boston dataset
boston = datasets.load_boston()

# Without any predefined fixed features
#-------------------------------------- 
data = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.Series(boston.target)

final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw = fibe(data, target, verbose=True)
print(final_features, validationPerformance)

# With predefined fixed features 
#-------------------------------
data = pd.DataFrame(boston.data, columns=boston.feature_names)
fixed = ['CRIM', 'ZN']
target = pd.Series(boston.target)

final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw = fibe(data, target, fixed_features=fixed, voting_strictness='both', verbose=True)
print(final_features, validationPerformance)


#======================================
#      Classification Example
#======================================

# Load UCI ML Breast Cancer Wisconsin (Diagnostic) dataset 
cancer = datasets.load_breast_cancer()

# Without any predefined fixed features
#-------------------------------------- 
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
target = pd.Series(cancer.target)

final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw = fibe(data, target, task_type='classification', verbose=True)
print(final_features, actual_score, predicted_score, validationPerformance)

# With predefined fixed features 
#-------------------------------
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
fixed = ['mean area', 'radius error']
target = pd.Series(cancer.target)

final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw = fibe(data, target, fixed_features=fixed, task_type='classification', voting_strictness='strict', verbose=True)
print(final_features, validationPerformance)

#============================================================
#      Classification Example with Imbalanced Data
#============================================================
# Load digit dataset
digit = datasets.load_digits()

# Making data imbalanced
data = pd.DataFrame(digit.data, columns=digit.feature_names)
target = digit.target

target = np.where(target < 4, 0, target) # Replacing all classes from 0 to 3 with 0
target = np.where(target > 1, 1, target) # Replacing all classes from 4 to 9 with 1
target = pd.DataFrame(target)
#print(target.value_counts())


# Without any predefined fixed features
#-------------------------------------- 
final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw = fibe(data, target, task_type='classification', balance=True, model_name='consensus', metric='F1-score', voting_strictness='strict', maxIter=1, verbose=True)
print(final_features, validationPerformance)

# With predefined fixed features 
#-------------------------------
fixed = ['pixel_2_1', 'pixel_2_2']
final_features, subjectList, actual_score, predicted_score, validationPerformance, dfw = fibe(data, target, fixed_features=fixed, task_type='classification', balance=True, metric='F1-score', voting_strictness='strict', maxIter=1, verbose=True)
print(final_features, validationPerformance)












# ----------- Ignore Below -------------
'''
df = pd.read_csv('/home/ch225256/iq_prediction/IQ_prediction/chd_fibe/chd_all_features_scores_standardized.csv') # Load your own features/targets


feature_df = pd.concat([df.iloc[:, 0:18], df.iloc[:, 33:]], axis=1)
feature_df = feature_df.drop(["ID", "site"], axis=1)
#feature_df = feature_df.drop(["ID", "site", "ageatMRI", "sex"], axis=1)
score_df = df[["FullScaleIQ"]]
#fixed=['ageatMRI', 'sex']
#fixed = df[["ageatMRI", "sex"]]
#fixed.drop(10, inplace=True)
#fixed = fixed.to_numpy()



final_features, actual_score, predicted_score, validationPerformance = fibe(feature_df, score_df, voting_strictness='strict', verbose=False)
print(final_features, actual_score, predicted_score, validationPerformance)
'''

'''# Fit data to a model with the selected features and predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

kf5 = KFold(n_splits = 5, shuffle = False)
actual_score = []
predicted_score = []

for inner_fold in kf5.split(feature_df): 
    training = feature_df.iloc[inner_fold[0]]
    validation = feature_df.iloc[inner_fold[1]]
    
    tscore = score_df.iloc[inner_fold[0]]
    vscore = score_df.iloc[inner_fold[1]]

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(training, tscore.values.ravel())
    y_pred = model.predict(validation)
    
    # Calculate the mean absolute error for the validation set
    actual_score.append(vscore.values.ravel().tolist())
    predicted_score.append(y_pred.tolist())
    
actual = [item for sublist in actual_score for item in sublist]
predicted = [item for sublist in predicted_score for item in sublist]
print(actual, predicted)'''
