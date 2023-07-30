from tokenize import Double
from typing import List,Callable
import math

from numpy import matrix

vector1 = [2,3]
vector2 = [2,7]
def addVectors(v, w):
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]
def vector_sum(vectors):
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]
def subtractVectors(vectorOne,vectorTwo):
    return [v-w for v,w in zip(vectorOne,vectorTwo)]
    
def scalarMultiply(c: float, v: List[float]) -> List[float]:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vectorMeans(vectors: List[List [float]]) -> List[float]:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalarMultiply(1/n, vector_sum(vectors))

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
