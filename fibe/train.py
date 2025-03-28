# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from .types import Metric, TaskType

import os
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from . import loss_estimation
from . import balance_data


def train(
        max_iter: int,
        n_fold: int,
        feature_df: pd.DataFrame,
        score_df: pd.DataFrame,
        shuffle_flag: bool,
        random_seed: int,
        specialist_features: pd.DataFrame,
        task_type: TaskType,
        balance: bool,
        model_name: str,
        model: BaseEstimator,
        metric: Metric,
        tolerance: float,
        max_features: int,
        save_intermediate: bool,
        output_dir: str,
        verbose=False
):
    max_iter = max_iter
    k_fold = KFold(n_splits=n_fold, shuffle=shuffle_flag, random_state=random_seed)

    frequency_of_features_selected_all_fold = []

    if verbose:
        print(f"Total Number of Features to Traverse in each fold: {len(feature_df.columns)}")
        print(f"Tolerance is selected: {tolerance}")
        print(f"Maximum number of Features allowed under tolerance, if not best: {max_features}\n")

    for idx_folder, outer_fold in enumerate(k_fold.split(feature_df)):
        print("\n=================================================================================\n")
        flag_FI = 0
        flag_BE = 0

        train_val_df = feature_df.iloc[outer_fold[0]]
        train_val_score_df = score_df.iloc[outer_fold[0]]
        if len(specialist_features) != 0:
            train_specialist = specialist_features.iloc[outer_fold[0]]

        idx_folder = idx_folder + 1

        selected_features = []
        best_features = []

        if task_type == 'regression':
            lowest_error = float('inf')
            error_to_fill = lowest_error  # newadd
        else:
            highest_accuracy = float('-inf')
            error_to_fill = highest_accuracy  # newadd

        for i in range(max_iter):
            noFeat_tolerance = 0
            exit_flag = 0
            for q in range(len(feature_df.columns)):
                if verbose:
                    print(
                        f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}]", flush=True)
                temp_error = []
                for feature in feature_df.columns:
                    if feature in selected_features:
                        temp_error.append(error_to_fill)
                        continue

                    # Add the new feature to the selected features
                    temp_features = selected_features + [feature]
                    inner_error = []

                    for inner_fold in k_fold.split(train_val_df):
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
                                df_feature, df_score = balance_data.balance_data(
                                    pd.concat([train_spec, df_feature], axis=1), df_score)
                                df_val_feature, df_val_score = balance_data.balance_data(
                                    pd.concat([valid_spec, df_val_feature], axis=1), df_val_score)
                            else:
                                df_feature, df_score = balance_data.balance_data(
                                    df_feature, df_score)
                                df_val_feature, df_val_score = balance_data.balance_data(
                                    df_val_feature, df_val_score)
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
                                inner_error.append(loss_estimation.loss_estimation(
                                    metric, df_val_score, y_pred))
                        else:
                            # Train a Regressor with the selected features and predict
                            model.fit(df_feature, df_score.values.ravel())
                            y_pred = model.predict(df_val_feature)

                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation.loss_estimation(
                                metric, df_val_score, y_pred))

                    temp_error.append(np.mean(inner_error))  # newadd

                    if verbose:
                        if task_type == 'regression':
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] -> Feature Added: {feature} | Error Found: {np.mean(inner_error)}", flush=True)
                        else:
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] -> Feature Added: {feature} | Accuracy Found: {np.mean(inner_error)}", flush=True)

                if task_type == 'regression':
                    if np.min(temp_error) <= lowest_error * (1 + tolerance):
                        selected_features.append(feature_df.columns[np.argmin(temp_error)])

                        if np.min(temp_error) > lowest_error:
                            noFeat_tolerance += 1
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] -- {noFeat_tolerance} feature(s) allowed under tolerance, which is not better than the last best.")
                            if noFeat_tolerance == max_features:
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] -- Number of allowed features under tolerance is met. Exiting from further FI | Best Features remain: {best_features}")
                                exit_flag = 1
                                break

                        if np.min(temp_error) < lowest_error:
                            lowest_error = np.min(temp_error)
                            best_features = copy.deepcopy(selected_features)
                            noFeat_tolerance = 0

                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] - Traversal over all features finished | Best Error ({metric}): {lowest_error:.4f} | Current Error ({metric}): {np.min(temp_error):.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] - Traversal over all features finished | Best Error ({metric}): {lowest_error:.4f} | Current Error ({metric}): {np.min(temp_error):.4f} | Selected Features: {selected_features}", flush=True)
                    elif np.min(temp_error) > lowest_error * (1 + tolerance) or exit_flag == 1:
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + best_features
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] No additional feature improves performance beyond tolerance | Best Features: {all_feat} | Starting BE..", flush=True)
                            else:
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] No additional feature improves performance beyond tolerance | Best Features: {best_features} | Starting BE..", flush=True)
                        flag_FI = 1
                        break
                else:
                    if np.max(temp_error) >= highest_accuracy * (1 - tolerance) and exit_flag == 0:
                        selected_features.append(feature_df.columns[np.argmax(temp_error)])

                        if np.max(temp_error) < highest_accuracy:
                            noFeat_tolerance += 1
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] -- {noFeat_tolerance} feature(s) allowed under tolerance, which is not better than the last best.")
                            if noFeat_tolerance == max_features:
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] -- Number of allowed features under tolerance is met. Exiting from further FI | Best Features remain: {best_features}")
                                exit_flag = 1
                                break

                        if np.max(temp_error) > highest_accuracy:
                            highest_accuracy = np.max(temp_error)
                            best_features = copy.deepcopy(selected_features)
                            noFeat_tolerance = 0

                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] - Traversal over all features finished | Best Prediction ({metric}): {highest_accuracy:.4f} | Currenct Prediction ({metric}): {np.max(temp_error):.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] - Traversal over all features finished | Best Prediction ({metric}): {highest_accuracy:.4f} | Currenct Prediction ({metric}): {np.max(temp_error):.4f} | Selected Features: {selected_features}", flush=True)

                    elif np.max(temp_error) < highest_accuracy * (1 - tolerance) or exit_flag == 1:
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + best_features
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] No additional feature improves performance beyond tolerance | Best Features: {all_feat} | Starting BE..", flush=True)
                            else:
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | FI | Traversal: {q + 1}] No additional feature improves performance beyond tolerance | Best Features: {best_features} | Starting BE..", flush=True)
                        flag_FI = 1
                        break

            selected_features = copy.deepcopy(best_features)
            if len(selected_features) == 1:
                if verbose:
                    print(
                        f"[Fold: {idx_folder} | Iter: {i + 1} | -- | Traversal: -] Since there is only one feature selected in FI, skipping BE and next iterations.", flush=True)
                break

            # Backward Elimination
            # selected_features_ = copy.deepcopy(selected_features)

            for q in range(len(selected_features)):
                if verbose:
                    print(f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] NOTE: You may see less number of features traversed over in BE, missing ones were under tolerance but not the best.", flush=True)
                temp_error = []
                for feature in selected_features:  # changed
                    # remove a feature from the selected features
                    temp_features = copy.deepcopy(selected_features)
                    if len(temp_features) == 1:
                        continue
                    temp_features.remove(feature)
                    inner_error = []

                    for inner_fold in k_fold.split(train_val_df):
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
                                df_feature, df_score = balance_data.balance_data(
                                    pd.concat([train_spec, df_feature], axis=1), df_score)
                                df_val_feature, df_val_score = balance_data.balance_data(
                                    pd.concat([valid_spec, df_val_feature], axis=1), df_val_score)
                            else:
                                df_feature, df_score = balance_data.balance_data(
                                    df_feature, df_score)
                                df_val_feature, df_val_score = balance_data.balance_data(
                                    df_val_feature, df_val_score)
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
                                inner_error.append(loss_estimation.loss_estimation(
                                    metric, df_val_score, y_pred))
                        else:
                            # Train a Regressor with the selected features and predict
                            model.fit(df_feature, df_score.values.ravel())
                            y_pred = model.predict(df_val_feature)

                            # Calculate the mean absolute error for the validation set
                            inner_error.append(loss_estimation.loss_estimation(
                                metric, df_val_score, y_pred))

                    temp_error.append(np.mean(inner_error))  # newadd

                    if verbose:
                        if task_type == 'regression':
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] <- Feature Removed: {feature} | Error Found: {np.mean(inner_error)}", flush=True)
                        else:
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] <- Feature Removed: {feature} | Accuracy Found: {np.mean(inner_error)}", flush=True)

                if task_type == 'regression':
                    if np.min(temp_error) < lowest_error:
                        lowest_error = np.min(temp_error)
                        selected_features.remove(
                            selected_features[np.argmin(temp_error)])  # changed
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] - Traversal over all features finished | {metric}: {lowest_error:.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] - Traversal over all features finished | {metric}: {lowest_error:.4f} | Selected Features: {selected_features}", flush=True)
                    else:
                        flag_BE = 1
                        if verbose:
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] No removal of additional feature improves performance | Selected Features: {selected_features}", flush=True)
                        break
                else:
                    if np.max(temp_error) > highest_accuracy:
                        highest_accuracy = np.max(temp_error)
                        selected_features.remove(
                            selected_features[np.argmax(temp_error)])  # changed
                        if verbose:
                            if len(specialist_features) != 0:
                                all_feat = list(specialist_features.columns) + selected_features
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] - Traversal over all features finished | {metric}: {highest_accuracy:.4f} | Selected Features: {all_feat}", flush=True)
                            else:
                                print(
                                    f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] - Traversal over all features finished | {metric}: {highest_accuracy:.4f} | Selected Features: {selected_features}", flush=True)
                    else:
                        flag_BE = 1
                        if verbose:
                            print(
                                f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] No removal of additional feature improves performance | Selected Features: {selected_features}", flush=True)
                        break

            if flag_FI and flag_BE:
                if verbose:
                    print(
                        f"[Fold: {idx_folder} | Iter: {i + 1} | -- | Traversal: -] Since no addition or removal of any features improves performance, skipping next iterations.", flush=True)
                break

            if i == 0:
                f_set_1 = copy.deepcopy(selected_features)
            else:
                f_set = copy.deepcopy(selected_features)
                if sorted(f_set_1) == sorted(f_set):
                    if verbose:
                        print(
                            f"[Fold: {idx_folder} | Iter: {i + 1} | BE | Traversal: {q + 1}] Selected features in this iteration did not change from the previous iteration. So quiting further iterations.", flush=True)
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
                    with open(os.path.join(output_dir, f"Selected_features_at_fold_{idx_folder}.txt"), "w") as f:
                        f.write(f"{selected_features}")
                else:
                    raise PermissionError(
                        f"You do not have write permission to directory '{output_dir}'")

        # saving features selected across all outer folds
        frequency_of_features_selected_all_fold += [selected_features]

        # print(frequency_of_features_selected_all_fold)
    # feature_counts = Counter(frequency_of_features_selected_all_fold)
    # return feature_counts
    return frequency_of_features_selected_all_fold
