import random
from typing import List, TypeVar,Tuple

X=TypeVar("X")#generic data type to represent input
Y=TypeVar("Y")#generic data type to represent output

def splitData(data : List[X], percent : float)->Tuple[List[X],List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    dataCopy = data[:]
    random.shuffle(dataCopy)
    cutIndex = int(len(dataCopy)*percent)
    #return everything from start to cut and then everything after that as a tuple
    return dataCopy[:cutIndex],dataCopy[cutIndex:]

data = [n for n in range(10)]
print(splitData(data,0.2))

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    
    #split into 2 x lists, one to train other to test and same with y
    trainX,testX=splitData(data=xs,percent=1-test_pct)
    trainY,testY=splitData(data=ys,percent=1-test_pct)
    return(trainX,testX,trainY,testY)


xs = [x for x in range(1000)]  # xs are 1 ... 1000
ys = [2 * x for x in xs]       # each y_i is twice x_i
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

# Check that the proportions are correct
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250