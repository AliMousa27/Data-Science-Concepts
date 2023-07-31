import random
from typing import List, TypeVar,Tuple

T=TypeVar("T")#generic data type

def splitData(data : List[T], percent : float)->Tuple[List[T],List[T]]:
    """Split data into fractions [prob, 1 - prob]"""
    dataCopy = data[:]
    random.shuffle(dataCopy)
    cutIndex = int(len(dataCopy)*percent)
    #return everything from start to cut and then everything after that as a tuple
    return dataCopy[:cutIndex],dataCopy[cutIndex:]


data = [n for n in range(10)]
print(splitData(data,0.2))