# -*- coding: utf-8 -*-

"""Different ways to calculate correlation between edge-types."""

import math

from scipy import spatial, stats

__all__ = [
    'wilcoxon_test',
    'entropy_test',
    'spearmanr_test',
    'pearsonr_test',
]


def wilcoxon_test(v1, v2) -> float:  # original metric: the smaller the more similar
    statistic, _ = stats.wilcoxon(v1, v2)
    if statistic != statistic:
        statistic = 0
    return 1 / (math.sqrt(statistic) + 1)


def entropy_test(v1, v2) -> float:  # original metric: the smaller the more similar
    result = stats.entropy(v1, v2)
    if result != result:
        result = 0
    return result


def spearmanr_test(v1, v2) -> float:  # original metric: the larger the more similar
    correlation, _ = stats.mstats.spearmanr(v1, v2)
    if correlation != correlation:
        correlation = -1
    return sigmoid(correlation)


def pearsonr_test(v1, v2) -> float:  # original metric: the larger the more similar
    pearsonr = stats.mstats.pearsonr(v1, v2)
    if pearsonr != pearsonr:
        pearsonr = -1
    return sigmoid(pearsonr)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def cos_test(v1, v2) -> float:
    return 1 - spatial.distance.cosine(v1, v2)


def standardization(x: float) -> float:
    return (x + 1) / 2


def relu(x: float) -> float:
    return (abs(x) + x) / 2
