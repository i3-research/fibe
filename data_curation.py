
# Copyright (c) Ankush Kesri/Yangming Ou, Boston Children's Hospital/Harvard Medical School, 2024
# email: Ankush.Kesri@childrens.harvard.edu, Yangming.Ou@childrens.harvard.edu

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


import random
import pandas as pd
import numpy as np

def data_curation(feature_df, unknown_value = 'unknown', keyword = None, thrs1 = 0.4, thrs2 = 5, thrs3 = 0.8):
    
  '''
  feature_df    :   Input pandas df that will be curated
  unknown       :   User specified value equivalent to 'NaN' or unknown value in the dataframe
  keyword       :   User specified keyword or a list of keywords if present in feature name, feature will be dropped
  thrs1         :   Threshold for missing values tolerance
  thrs2         :   Threshold for the ratio of two classes for binary feature (to drop highly imbalanced binary features)
  thrs3         :   Threshold for the ratio of top two categories sum count to the total count of the feature values (to combined rest of the categories into third category)


  returns
  feature_df    :   Output curated pandas df
  drop_log      :   A dictionary of dropped features as keys and reason for dropping as value
  mapping_list  :   A list of lists for mapping record for column name, original value and mapped value: [col, key, value] in order
  imputation_log:   A list of lists for track of imputed category or median value e.g. [col, 'categorical', chosen_cat] in order

  '''

  random.seed(42) #(for reprodicibility in part C)  
    
  # For log files
  drop_log = {} # For A: Exclusion criteria
  mapping_list = [] # For B: Recoding multi-class features
  imputation_log = [] # For C: Filling Minor missing data

  # Converting all values to lowercase strings and then to boolean
  try:
    feature_df = feature_df.replace({'true': True, 'false': False})
  except:
    pass

  # List of keywords if found in column name, will drop the column in Step 3
  keywords = ['id', 'note', 'complete', 'comment', 'description', 'date', 'time', 'timestamp', 'year', 'month', 'day', 'hour', 'minute']
  if keyword != None:
    keywords.append(keyword)

  #####################( A: Exclusion criteria )################################

  for col in feature_df.columns:
      
    original = feature_df[col].copy()  # Keep a copy of the original data
    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')  # Try converting to numeric
    
    # Check if the column conversion was successful
    if feature_df[col].isna().all():  # If all values became NaN, conversion failed
      feature_df[col] = original  # Revert the changes
  
  
    # Step 1: Handling Missing Values
    # Step 1a: Replacing all unknown values with np.nan
    if unknown_value in feature_df[col].values:
      feature_df[col].replace([unknown_value], np.nan, inplace = True)

    # Step 1b: Counting nan values and dropping column if more than 40% data is missing
    na_count = feature_df[col].isna().sum()
    if na_count > thrs1*len(feature_df):                                                            # Threshold 1
      feature_df.drop(col, axis=1, inplace = True)
      drop_log[col] = 'Reason 1: More than 40% missing data'
      continue

    # Step 2: Handling Imbalanced distributions
    # Step 2a: Dropping column with no variance (only 1 value for all cases)
    if feature_df[col].nunique() == 1:
      feature_df.drop(col, axis=1, inplace = True)
      drop_log[col] = 'Reason 2: No Variance, only 1 value for all cases'
      continue

    # Step 2b: Dropping column if binary and the ratio is 5:1 (default) or greater
    if feature_df[col].nunique() == 2:
      temp_value_counts = feature_df[col].value_counts()
      major_class_key = temp_value_counts.keys()[0]
      minor_class_key = temp_value_counts.keys()[1]

      if (temp_value_counts[major_class_key] / temp_value_counts[minor_class_key]) >=thrs2:          # Threshold 2
        feature_df.drop(col, axis=1, inplace = True)
        drop_log[col] = 'Reason 3: Binary column with high imbalance'
        continue

    # Step 3: Handling Illegitimate Features
    if any(kw in col.lower() for kw in keywords) and 'age' not in col.lower():
      feature_df.drop(col, axis=1, inplace = True)
      drop_log[col] = 'Reason 4: Illegitimate Feature, maybe date, time or description string'
      continue


  #################( B: Recoding multi-class features )#########################
  for col in feature_df.columns:

    unique_values = feature_df[col].nunique()
    # Handling Binary columns (Columns with only two categories)
    if unique_values == 2:

      value_counts = feature_df[col].value_counts()
      major_class_key = value_counts.keys()[0]
      minor_class_key = value_counts.keys()[1]

      if feature_df[col].apply(lambda x: isinstance(x, bool)).any():
        feature_df[col] = feature_df[col].fillna(False)
        feature_df[col] = feature_df[col].astype(int)
        mapping_list.append([col, 'True', 1])
        mapping_list.append([col, 'False', 0])
        continue

      
      if major_class_key == 0 or major_class_key == 1: #incase values are 1 and 0 instead of true and false
        continue
      binary_mapping = {major_class_key : 0, minor_class_key : 1}

      # Applying mapping to the binary column
      feature_df[col] = feature_df[col].map(binary_mapping)
      # Adding column name, original value and mapped value in nested list
      for key, value in binary_mapping.items():
        mapping_list.append([col, key, value])
      continue

    # Handling categorical features
    if unique_values > 2:

      # Handling features where minor classes will need to be combined if first two major classes occupy more than 80% count
      value_counts = feature_df[col].value_counts()
      total_count = value_counts.sum()
      count_top2cat = value_counts.iloc[:2].sum()

      # Getting names of major classes for mapping
      major_class_key = value_counts.keys()[0]
      minor_class_key = value_counts.keys()[1]

      if count_top2cat >= thrs3*total_count:                                                        # Threshold 3
        cat_list = value_counts.keys().to_list()[:2] # only keeping two major classes in list       # Threshold: Will need manual change for required categories
        feature_df[col] = feature_df[col].apply(lambda x: x if x in cat_list else 'Other')

        mapping = {major_class_key : 0, minor_class_key : 1, 'Other' : 2}

        # Applying mapping to the column
        feature_df[col] = feature_df[col].map(mapping)
        # Adding column name, original value and mapped value in nested list
        for key, value in mapping.items():
          mapping_list.append([col, key, value])
        continue

      # Ensuring that categorical values with predefined numeric categories are kept as it is
      if feature_df[col].dropna().apply(lambda x: isinstance(x, (int,float))).any() and unique_values < 30:
        continue

      if unique_values <= 30:  # Upper bound for reasonable/realistic amount of categories per feature (trust me, it took several rounds of debugging)
        value_counts = feature_df[col].value_counts()
        mapping = {category: idx + 1 for idx, category in enumerate(value_counts.index)}

        # Applying mapping to the column
        feature_df[col] = feature_df[col].map(mapping)
        # Adding column name, original value and mapped value in nested list
        for key, value in mapping.items():
          mapping_list.append([col, key, value])
        continue


  #################( C: Filling Minor missing data )############################

	def var_type(column):
	    unique_vals = column.nunique()
	    if pd.api.types.is_numeric_dtype(column):
	        return 'continuous' if unique_vals > 8 else 'categorical'
					# 8 = arbitrary value, Keeping it low, imputing categorical variable (integer) with median won't harm but continuos with random value instead of median might!
	    else:
	        return 'categorical'
	
	for col in feature_df.columns:
	    feature_type = var_type(feature_df[col])
	    missing_before = feature_df[col].isnull().sum()
	
	    if feature_type == 'continuous':
	        if missing_before > 0:
	            median_val = feature_df[col].median()
	            feature_df[col] = feature_df[col].fillna(median_val)
	            imputation_log.append([col, 'continuous', median_val])
	
	    elif feature_type == 'categorical':
	        unique_cats = feature_df[col].dropna().unique()
	
	        if missing_before > 0:
	            if len(unique_cats) == 2:
	                major_cat = feature_df[col].mode()[0]
	                feature_df[col] = feature_df[col].fillna(major_cat)
	                imputation_log.append([col, 'categorical-binary', major_cat])
	            else:
	                chosen_cat = random.choice(unique_cats)
	                feature_df[col] = feature_df[col].fillna(chosen_cat)
	                imputation_log.append([col, 'categorical', chosen_cat])

  return feature_df, drop_log, mapping_list, imputation_log



def log_files_generator(drop_log, mapping_list, imputation_log, output_dir):
  '''
  Checks if output_dir was provided by user in fibe function
  Takes output from data_curation and creates and saves excel files for drop_log, mapping_list, imputation_log
  '''
  
  if output_dir == None:
    dropped_features_log = pd.DataFrame(list(drop_log.items()), columns=['Feature', 'Reason for dropping'])
    dropped_features_log.to_excel('dropped_features_log.xlsx', index = False)
    
    mapping_list_df = pd.DataFrame(mapping_list, columns = ['Feature', 'Old', 'Mapped'])
    mapping_list_df.to_excel('mapping_list.xlsx', index = False)
    
    imputation_log_df = pd.DataFrame(imputation_log, columns = ['Feature', 'Type', 'Imputation Value'])
    imputation_log_df.to_excel('imputation_log.xlsx', index = False)
    
  else:
    if not output_dir.endswith('/'):
      output_dir += '/'
      
    dropped_features_log = pd.DataFrame(list(drop_log.items()), columns=['Feature', 'Reason for dropping'])
    dropped_features_log_file_path = f"{output_dir}dropped_features_log.xlsx"
    dropped_features_log.to_excel(dropped_features_log_file_path, index = False)
    
    mapping_list_df = pd.DataFrame(mapping_list, columns = ['Feature', 'Old', 'Mapped'])
    mapping_list_file_path = f"{output_dir}mapping_list.xlsx"
    mapping_list_df.to_excel(mapping_list_file_path, index = False)
    
    imputation_log_df = pd.DataFrame(imputation_log, columns = ['Feature', 'Type', 'Imputation Value'])
    imputation_log_file_path = f"{output_dir}imputation_log.xlsx"
    imputation_log_df.to_excel(imputation_log_file_path, index = False)      
    
  

