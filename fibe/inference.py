# -*- coding: utf-8 -*-


from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
import pandas as pd

from .types import Metric, TaskType

from . import loss_estimation


def inference(
        final_features: list[str],
        nFold: int,
        feature_df: pd.DataFrame,
        score_df: pd.DataFrame,
        shuffle_flag: bool,
        random_seed: int,
        specialist_features: pd.DataFrame,
        balance,
        model_name: str,
        model: BaseEstimator,
        metric: Metric,
        task_type: TaskType,
        probability: bool
):

    k_fold = KFold(n_splits=nFold, shuffle=shuffle_flag, random_state=random_seed)
    val_performance_by_fold = []
    actual_score = []
    predicted_score = []
    subjects = []
    predicted_probs = []

    for infer_fold in k_fold.split(feature_df):
        train_val_df = feature_df.iloc[infer_fold[0]]
        test_df = feature_df.iloc[infer_fold[1]]

        # Getting the subject IDs for the test set
        test_subjects = list(infer_fold[1])

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
                    prob_class_1 = y_pred_proba[:, 1]  # probability of class 1
                    probabilities.append(prob_class_1)

            if probability == True:
                # Averaging the probabilities from the 3 models
                consensus_prob = [(a + b + c) / 3 for a, b,
                                  c in zip(probabilities[0], probabilities[1], probabilities[2])]
                predicted_probs.append(consensus_prob)

            if task_type == 'regression':
                consensus_pred = [(a + b + c) / 3 for a, b,
                                  c in zip(predictions[0], predictions[1], predictions[2])]
            elif task_type == 'classification':
                def majority_vote(a, b, c):
                    return 1 if a + b + c > 1 else 0
                consensus_pred = [majority_vote(a, b, c) for a, b, c in zip(
                    predictions[0], predictions[1], predictions[2])]

            # Calculating the performance for the validation set
            performance = loss_estimation.loss_estimation(metric, df_val_s, consensus_pred)

            # Saving the actual, predicted scores, and subject IDs
            actual_score.append(df_val_s.values.ravel().tolist())
            predicted_score.append(consensus_pred)
            subjects.append(test_subjects)

            val_performance_by_fold.append(performance)
        else:
            # Fitting data to a model with the selected features and predict
            model.fit(df_f, df_s.values.ravel())
            y_pred = model.predict(df_val_f)

            if probability == True and task_type == 'classification':
                y_pred_proba = model.predict_proba(df_val_f)
                prob_class_1 = y_pred_proba[:, 1]  # probability of class 1
                predicted_probs.append(prob_class_1)

            # Calculating the performance for the validation set
            val_performance_by_fold.append(
                loss_estimation.loss_estimation(metric, df_val_s, y_pred))
            actual_score.append(df_val_s.values.ravel().tolist())
            predicted_score.append(y_pred.tolist())
            subjects.append(test_subjects)

    # Flattening the lists of actual scores, predicted scores, and subject IDs
    actual = [item for sublist in actual_score for item in sublist]
    predicted = [round(item, 2) for sublist in predicted_score for item in sublist]
    subjects = [item for sublist in subjects for item in sublist]

    if probability == True:
        predicted_probs = [round(item, 2) for sublist in predicted_probs for item in sublist]
        return subjects, actual, [predicted] + [predicted_probs], val_performance_by_fold
    else:
        return subjects, actual, predicted, val_performance_by_fold
