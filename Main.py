from tokenize import Double
from typing import List,Callable
import math

from numpy import matrix

vector1 = [2,3]
vector2 = [2,7]

def addVectors(*vectors: List)->List:
    return [sum(dimension) for dimension in zip(*vectors)]

def subtractVectors(vectorOne,vectorTwo):
    return [v-w for v,w in zip(vectorOne,vectorTwo)]
    
def scalarMultiply(vector: List,scalar:Double)->List: 
    return [scalar * dimension for dimension in vector]

def vectorMeans(*vectors: List)->List:
    return [sum(dimension)/len(dimension) for dimension in zip(*vectors)]

def dotProduct(vectorOne,vectorTwo) -> Double:
    return sum([v*w for v,w in zip(vectorOne,vectorTwo)])

def sumOfSquares(vector:list)->List:
    #we are just multipying v by itself then summing the result
    return dotProduct(vector,vector)

def magnitude(vector:list)->float:
    #magnitude is defined as the sqrt of the sum of squaraes literally lol
    return math.sqrt(sumOfSquares(vector))

def distanceBetweenVectors(vectorOne:List,vectorTwo:List) -> Double:
    #defined as the square root of the sum of squares of the idffernec ebteween each compoenent 
    #of each vector
    return magnitude(subtractVectors(vectorOne,vectorTwo))


A =[[1,2],
    [2,4]]

def matrixShape(A: List[List]):
    rows = len(A)
    columns = len(A[0])
    return rows,columns

def get_row(A, i: int):
    """Returns the i-th row of A (as a Vector)"""
    return A[i]             # A[i] is already the ith row

def get_column(A, j: int):
    """Returns the j-th column of A (as a Vector)"""
    return [row[j]          # jth element of current row
            for row in A]   # for each row 
    
    
    #callable is a function that takes 2 integers and returns a float so i am passing in a fucking method
def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]):
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             # given i, create a list with what the function returns and give it i and j
             for j in range(num_cols)]  #   [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # create one list for each i
    
def identity(n:int):
    return make_matrix(5,5,lambda i,j: 1 if i==j else 0)

print(identity(5))