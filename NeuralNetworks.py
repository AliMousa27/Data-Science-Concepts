from vectors import Vector, dot
import math
from typing import List
#function to determine if a perceptron should fire
def stepFunction(x : float) -> float:
    return 1 if x>=0 else 0

#computes the weighted sum of inputs x
def perceptronOutput(weights: Vector, bias: float, x:Vector) -> float:
    #returns 1 if the perceptron should fire 0 if it shouldnt
    return stepFunction(dot(x,weights)+bias)

def sigmod(x:float)->float:
    return 1/(1+math.exp(-x))

def neuronOutput(weights: Vector, inputs: Vector)->float:
    """function to out put the calculation between the weighted sum inputted into sigmoid. 
    Bias is the last element of the weights and inputs will include a 1"""
    return sigmod(dot(weights,inputs))

 
""" Neural network is a 3d list,
Neuron = list of vector weights 
layer= list of neurons
Network = list of layers 
So a network is a 3rd list 
"""
def feedForward(neuralNetwork: List[List[Vector]],
                inputVector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """






def main():
    #OR gate test using the perceptrons
    orWeights = [2., 2]
    orBias = -1.
    assert perceptronOutput(orWeights, orBias, [1, 1]) == 1
    assert perceptronOutput(orWeights, orBias, [0, 1]) == 1
    assert perceptronOutput(orWeights, orBias, [1, 0]) == 1
    assert perceptronOutput(orWeights, orBias, [0, 0]) == 0
    
    #AND gate test using the perceptrons
    andWeights = [2., 2]
    andBias = -3.
    assert perceptronOutput(andWeights, andBias, [1, 1]) == 1
    assert perceptronOutput(andWeights, andBias, [0, 1]) == 0
    assert perceptronOutput(andWeights, andBias, [1, 0]) == 0
    assert perceptronOutput(andWeights, andBias, [0, 0]) == 0

    notWeights =[-2]
    notBias = 1
    assert perceptronOutput(notWeights, notBias, [1])==0
    assert perceptronOutput(notWeights, notBias, [0])==1
if __name__ == "__main__": main()