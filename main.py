import pandas as pd
from fibe_functions import fibe
import numpy as np


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

#final_features = ['deltaAGE=median-actual', 'chr_var', 'DAD_ED_LEV_STD', 'mri_all_PC4', 'mri_all_PC10', 'rsBOLD_all_aparc2_gordon_network_corr_PC37']
#print(list(fixed.columns) + final_features)











#column_names_list = list(feature_df.columns)
#feature_df = feature_df.drop('ageatMRI', axis=1)
# Convert DataFrame to NumPy array
#numpy_array = feature_df.to_numpy()
#feature_df = feature_df.values.tolist()
#if type(numpy_array) is np.ndarray:
#  df = pd.DataFrame(data=numpy_array, columns=column_names_list)
#  print(df)

#print(feature_df)
#print(score_df)