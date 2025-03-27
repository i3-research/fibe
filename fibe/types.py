# -*- coding: utf-8 -*-

from typing import Literal

type ModelNameRegression = Literal[
    'linearSVR',
    'gaussianSVR',
    'RegressionForest',
    'AdaBoostDT',
    'AdaBoostSVR',
    'consensus',
]

type ModelNameClassification = Literal[
    'linearSVC',
    'gaussianSVC',
    'RandomForest',
    'AdaBoostDT',
    'AdaBoostSVC',
    'consensus',
]

type ModelName = ModelNameRegression | ModelNameClassification

type MetricRegression = Literal['MAE', 'MAPE']

type MetricClassification = Literal['Accuracy', 'F1-score', 'binaryROC']

type Metric = MetricRegression | MetricClassification

type TaskType = Literal['regression', 'classification']

type StrictType = Literal['strict', 'loose', 'conditional', 'weighted', 'union']
