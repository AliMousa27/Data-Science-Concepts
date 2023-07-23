from numpy import mean
import vectors
import math,random

from typing import Callable,List
#returns the gradient at a given point
def differenceQuotient(function: Callable[[float],float],point:float,changeAmount:float) -> float:
    return (function(point+changeAmount)-function(point))/changeAmount
def func(num):
    return num*num

def paritalDifferenceQuotient(function: Callable[[List[float]],float],
                              vector:List[float],
                              index:int,#index of the ith elemnt of the vector that we want to change
                              changeAmount:float) -> float:
    #first copy all the elemnts into a new vector and add the change amount only if the elemnt is 
    # the i-th element
    increase=[dimension + (changeAmount if j==index else 0)#only add when we reach the ith element in the vector
              #that we want the chanve amount of
              for j,dimension in enumerate(vector)]
    return (function(increase)-function(vector))/changeAmount

#estimates the gradient for each and every variable of the funciton by iterating over them
def estimateGradient(f: Callable[[List[float]], float],
                      v: List[float],
                      h: float = 0.0001):
    return [paritalDifferenceQuotient(f, v, i, h)
            for i in range(len(v))]        
    


def gradient_step(v: List[float], gradient: List[float], step_size: float) -> List[float]:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    #multiply by the step size to no overshoot. if graident is small we take a small step
    #when gradient is 0 we stop moving 
    step = vectors.scalarMultiply(step_size, gradient)
    #then we add the step size to the original point to move towards max/min
    #then we add 0 basically to it if we reach max or min
    return vectors.addVectors(v, step)

def sum_of_squares_gradient(v: List[float]) -> List[float]:
    return [2 * v_i for v_i in v]


v = [random.uniform(-10, 10) for i in range(3)]
    
#should be a larger range to optimize the function
for epoch in range(1):
    grad = sum_of_squares_gradient(v)    # compute the gradient at v
    v = gradient_step(v, grad, -0.01)    # take a negative gradient step
    print(epoch, v)
    
#assert vectors.distanceBetweenVectors(v, [0, 0, 0]) < 0.001    # v should be close to 0


inputs = [(x, x * x + 5) for x in range(-50, 50)]
def linear_gradient(x: float, y: float, theta: List[float]) -> List[float]:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual)
    squared_error = error ** 2           # We'll minimize squared error
    grad = [2 * error * x*x, 2 * error]    # using its gradient.
    return grad

#start at random value
theta = [random.uniform(-1,1),random.uniform(-1,1)]
stepSize= -0.001
for epoch in range(100):
    meanGrad = vectors.vectorMeans([linear_gradient(x,y,theta) for x,y in inputs])
    #take step to optimize theta
    theta = gradient_step(theta,meanGrad,stepSize)
    print(epoch,theta)
    
    
'''for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x,y,theta)
    #take step to optimize theta
        theta = gradient_step(theta,grad,stepSize)
        print(epoch,theta)'''