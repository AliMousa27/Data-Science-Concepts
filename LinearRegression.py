
from Gradients import gradient_step
from vectors import Vector
from Statistics import correlation, de_mean,standard_deviation,mean
from typing import Tuple
from Statistics import num_friends_good, daily_minutes_good
import random
import tqdm
def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha
    when the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i


def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

from typing import Tuple

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Given two vectors x and y,
    find the least-squares values of alpha and beta
    """
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

# Should find that y = 3x - 5
assert least_squares_fit(x, y) == (-5, 3)


alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905


def total_sum_of_squares(y: Vector) -> float:
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330


def main():
    thetaGuess =[random.random(),random.random()]#take a random guess at first
    learningRate = -0.00001
    
    for _ in range(100000):
            m,c=thetaGuess
            #parital derivative of the squared error with respect to c
            gradientC = sum(2* error(c,m,xi,yi) for xi , yi in zip(num_friends_good,daily_minutes_good))
            #parital derivative of the squared error with respect to m
            gradientM = sum(2* error(c,m,xi,yi)*xi for xi , yi in zip(num_friends_good,daily_minutes_good))
            
            thetaGuess = gradient_step(thetaGuess,[gradientM,gradientC],learningRate)
            print(thetaGuess)
if __name__ == "__main__": main()