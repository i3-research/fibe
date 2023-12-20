# Import necessary libraries
import pandas as pd
from sklearn import datasets
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

final_features, validationPerformance = fibe(data, target, voting_strictness='strict', verbose=True)
print(final_features, validationPerformance)

# With predefined fixed features 
#-------------------------------
data = pd.DataFrame(boston.data, columns=boston.feature_names)
fixed=['CRIM', 'ZN']
target = pd.Series(boston.target)

final_features, validationPerformance = fibe(data, target, fixed_features=fixed, voting_strictness='strict', verbose=True)
print(final_features, validationPerformance)


#======================================
#      Classification Example
#======================================

# Load UCI ML Breast Cancer Wisconsin (Diagnostic) dataset dataset
cancer = datasets.load_breast_cancer()

# Without any predefined fixed features
#-------------------------------------- 
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
target = pd.Series(cancer.target)

final_features, validationPerformance = fibe(data, target, task_type='classification', voting_strictness='strict', verbose=True)
print(final_features, validationPerformance)

# With predefined fixed features 
#-------------------------------
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
fixed=['mean area', 'radius error']
target = pd.Series(cancer.target)

final_features, validationPerformance = fibe(data, target, fixed_features=fixed, task_type='classification', voting_strictness='strict', verbose=True)
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

final_features, validationPerformance = fibe(feature_df, score_df, model_name='AdaBoostSVR', voting_strictness='strict', verbose=False)
print(final_features, validationPerformance)
'''