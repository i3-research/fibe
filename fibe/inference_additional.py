# -*- coding: utf-8 -*-

import pandas as pd

from . import loss_estimation


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

    df_val_s = inference_score_df

    # Extracting subject IDs (e.g., sub1, sub2, sub3,...)
    inference_subjects = list(range(len(inference_data_df)))

    if model_name == 'consensus':
        predictions = []
        probabilities = []
        for one_model in model:
            model_ = model[one_model]

            # Fit data to a model with the selected features and predict
            model_.fit(df_val_f, score_df.values.ravel())
            y_pred = model_.predict(inference_data_df2)
            predictions.append(y_pred)

            if probability == True:
                y_pred_proba = model_.predict_proba(inference_data_df2)
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

        # Calculate the performance for the validation set
        performance = loss_estimation.loss_estimation(
            metric, df_val_s.values.ravel(), consensus_pred)

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
        return subjects, actual, [predicted] + [predicted_probs], valPerformanceByFold
    else:
        return subjects, actual, predicted, valPerformanceByFold
