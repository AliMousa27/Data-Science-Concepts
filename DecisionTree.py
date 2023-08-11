from typing import List,Any
import math
from collections import Counter


def entropy(classProbabilities: List[float]) -> float:
    '''classProbabilities is a list of p values where each p value is
        the proportion of hoiw much of our data belongs to that class
    '''
    #if statement to ignore data with no classes
    return sum([-p*math.log(p,2) for p in classProbabilities if p > 0])


#our data is a list of (input,label) here we compute the percent of each label
def classProbabilities(labels: List[Any]) -> List[float]:
    n= len(labels)
    #return a list of proprotons of each label
    return [p/n for p in Counter(labels).values()]
#returns the entropy of a given data set
def dataEntropy(labels: List[Any]) -> float:
    return entropy(classProbabilities(labels))