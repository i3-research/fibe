# -*- coding: utf-8 -*-

from sklearn import metrics

import numpy as np

from .types import Metric


def loss_estimation(metric: Metric, true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    if metric == 'MAE':
        mae = metrics.mean_absolute_error(true_values, predicted_values)
        return mae
    elif metric == 'MAPE':
        true_values_no_zero = np.where(true_values == 0, 1e-10, true_values)
        mape = np.mean(np.abs((true_values - predicted_values) / true_values_no_zero)) * 100
        return mape
    elif metric == 'Accuracy':
        accuracy = metrics.accuracy_score(true_values, predicted_values)
        return accuracy
    elif metric == 'binaryROC':
        cm = metrics.confusion_matrix(true_values, predicted_values)
        tn, fp, fn, tp = cm.ravel()
        epsilon = 1e-7
        sensitivity = round(tp / (tp + fn + epsilon), 2)
        specificity = round(tn / (tn + fp + epsilon), 2)
        binaryROC = ((1 - sensitivity) ** 2) + ((1 - specificity) ** 2)
        return binaryROC
    elif metric == 'F1-score':
        cm = metrics.confusion_matrix(true_values, predicted_values)
        tn, fp, fn, tp = cm.ravel()
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        f1_score = round(f1_score, 2)
        return f1_score
    else:
        raise ValueError("Unknown metric")
