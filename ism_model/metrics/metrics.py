from typing import List, Union


def Theil(prediction: List[Union[int, float]], real: List[Union[int, float]]) -> float:
    numerator = sum([(a + b)**2 for a, b in zip(prediction, real)])
    denominator = sum([a**2 + b**2 for a, b in zip(prediction, real)])
    return (numerator/denominator)**(0.5)


def RMSPE(prediction: List[Union[int, float]], real: List[Union[int, float]]) -> float:
    return (sum([((a-b)**2)/b for a, b in zip(prediction, real)])/len(real))**(0.5)
