from typing import List, Union
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np


def Theil(prediction: List[Union[int, float]], real: List[Union[int, float]]) -> float:
    numerator = sum([(a + b)**2 for a, b in zip(prediction, real)])
    denominator = sum([a**2 + b**2 for a, b in zip(prediction, real)])
    return (numerator/denominator)**(0.5)


def RMSPE(prediction: List[Union[int, float]], real: List[Union[int, float]]) -> float:
    return (sum([((a/30-b)**2)/(b**2) for a, b in zip(prediction, real)])/len(real))**(0.5)


def MAPE(prediction: List[Union[int, float]], real: List[Union[int, float]]):
    return mean_absolute_percentage_error(real, prediction)


def R2(prediction: List[Union[int, float]], real: List[Union[int, float]]):
    return r2_score(real, prediction)


def CORR(prediction: List[Union[int, float]], real: List[Union[int, float]]):
    return np.corrcoef(prediction, real)[0][1]
