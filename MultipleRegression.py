from typing import List,TypeVar, Callable,Tuple
from LinearRegression import total_sum_of_squares
from vectors import Vector, dot,vector_mean
import tqdm
from Gradients import gradient_step
import random
from Statistics import daily_minutes_good,standard_deviation

#for multiple regression, the yith value is xi multiplied by beta
#both of which are a vector so we multiply them using the dot product
def predict(x:Vector,beta:Vector) -> float:
    return dot(x,beta)

#these are our parameters. we try to find a vector beta that best fits the model. that is to 
#say we try to find the best fit paramers inside beta vector
[1,    # constant term
 49,   # number of friends
 4,    # work hours per day
 0]    # doesn't have PhD

def     error(x:Vector,beta:Vector,actualY:float)->float:
    return predict(x,beta)-actualY
def squaredError(x:Vector,beta:Vector,actualY:float)->float:
    return error(x,beta,actualY)**2

def squaredErrorGradient(x:Vector,actualY:float,beta:Vector)->Vector:
    err= error(x,beta,actualY)
    #return the graident of the squared error with respect to x
    return [2*err*xi for xi in x]

def leastSquaresFit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    #start at a random point with a vecotr of parameteres equal in length to the data
    theta = [random.random() for _ in xs[0]] 
    for _ in tqdm.trange(num_steps, desc="Least squares fit "):
        #iterate over the list with index start. 
        for start in range(0,len(xs),batch_size):
            batchXs = xs[start:start+batch_size]
            batchYs = ys[start:start+batch_size]
            #get the gradient which is the mean of all the gradient vectors for each data list
            gradient = vector_mean([squaredErrorGradient(xi,yi,theta) 
                                   for xi,yi in zip(batchXs,batchYs)])
            #update theta
            theta=gradient_step(theta,gradient,-learning_rate)
    return theta

learning_rate = 0.001
inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]
beta = leastSquaresFit(inputs, daily_minutes_good, learning_rate, 5000, 25)

def multipleRSquaredxs(xs : List[Vector], ys: List[float], beta: List[Vector]) -> List[Vector]:
    rss = sum([error(x,beta,y)**2 for x, y in zip(xs,ys)])
    return 1-rss/total_sum_of_squares(ys)

#generic types for data and statisitc
X= TypeVar("X")
Stat = TypeVar("Stat")

def bootStrapSample(data:List[X]) ->List[X]:
    #randomly samples from a data set with replacement, that is to say
    #some data poitns might get selcted multiple times while others might not be selected at all
    return [random.choice(data) for _ in data]


def bootStrapStatistic(data: List[X],
                        statsFN: Callable[[List[X]], Stat],
                        numsSamples: int) -> List[Stat]:
    """executes stats_fn on bootstrap samples from data in range of however many samples we have"""
    return[statsFN(bootStrapSample(data)) for _ in range(numsSamples)]

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
        '''returns a beta for a list of samples'''
        x_sample = [x for x, _ in pairs]
        y_sample = [y for _, y in pairs]
        beta = leastSquaresFit(x_sample, y_sample, learning_rate, 5000, 25)
        print("bootstrap sample", beta)
        return beta
bootStrapBetas = bootStrapStatistic(list(zip(inputs,daily_minutes_good)),estimate_sample_beta,5)

bootstrapSTDEV = [standard_deviation([beta[i] for beta in bootStrapBetas])
                                    for i in range(4)]

