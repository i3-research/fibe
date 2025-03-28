# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import utils as sklearn_utils


def balance_data(data: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
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
    df_minority_upsampled = sklearn_utils.resample(
        df_minority,
        replace=True,
        n_samples=df_majority.shape[0],
        random_state=123)

    # Combine majority class with upsampled minority class
    df_upsampled: pd.DataFrame = pd.concat([df_majority, df_minority_upsampled])

    # Separate features and target from the balanced dataframe
    feature_balanced = df_upsampled.drop('groundTruth', axis=1)
    class_balanced = df_upsampled['groundTruth']

    return feature_balanced, class_balanced
