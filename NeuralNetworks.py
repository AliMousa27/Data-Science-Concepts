import tqdm
from vectors import Vector, dot
import math
from typing import List
import random
from Gradients import gradient_step
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
    outputs: List[Vector] = []
    for layer in neuralNetwork:
        inputWithBias = inputVector + [1] # Add a constant.
        #make the calculation for each neuron in the layer
        currentOutPut =[neuronOutput(neuron,inputWithBias) for neuron in layer]
        #add it to the list of outputs
        outputs.append(currentOutPut)
        #then the input of the next layer is the output of the current layer
        inputVector = currentOutPut
        
    return outputs

def sqerrorGradients(network: List[List[Vector]],
                      inputVector: Vector,
                      targetVector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector, and a target vector,
    make a prediction and compute the gradient of the squared error
    loss with respect to the neuron weights.
    """
    # forward pass
    hiddenOuputs, outputs = feedForward(network,inputVector)
    
    # gradients with respect to output neuron pre-activation outputs
    #here the graidnet uses the chain rule as now we multiply the sigmoid derivatve by the error
    outputDeltas = [output * (1 - output) * (output - target)
                    for output, target in zip(outputs, targetVector)]
    
    # gradients with respect to output neuron weights
    outputGrads = [[outputDeltas[i] * hiddenOutput #multiply each output gradient with the hidden neuron weights
                    for hiddenOutput in hiddenOuputs + [1]]#add the bias constant
                    for i, outputNeuron in enumerate(network[-1])]#enumarate over the last layer in the network
    
    # gradients with respect to hidden neuron pre-activation outputs
    hiddenDeltas = [hiddenOutput * (1 - hiddenOutput) *#multiply the gradient of the hiddeninput sigmoid function 
    dot(outputDeltas, [n[i] for n in network[-1]])# by the sum of the loss gradient with the last layer of neurons in the network
    for i, hiddenOutput in enumerate(hiddenOuputs)]
    
    #2 pervious variables made use of the chain rule
    
    # gradients with respect to hidden neuron weights
    hiddenGrads = [[hiddenDeltas[i] * input for input in inputVector + [1]]
    for i, hiddenNeuron in enumerate(network[0])]
    return [hiddenGrads, outputGrads]
    
    
    
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
    
    xorNetwork = [
    # hidden layer
    [[20., 20, -30], # 'and' neuron
    [20., 20, -10]], # 'or' neuron
    #the before was 1 layer´which is hidden that preforms calcualtions to feed the output layer
    # output layer
    [[-60., 60, -30]]# '2nd input but not 1st input' neuron
    ] 

    #print(feedForward(xorNetwork, [1, 0])[-1][-1])
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]
    
    network = [ # hidden layer: 2 inputs -> 2 outputs
    [[random.random() for _ in range(2 + 1)], # 1st hidden neuron
    [random.random() for _ in range(2 + 1)]], # 2nd hidden neuron
    # output layer: 2 inputs -> 1 output
    [[random.random() for _ in range(2 + 1)]] # 1st output neuron
    ]

    lr =1
    for epoch in tqdm.trange(20000, desc="neural net for xor"):
        for x,y in zip(xs,ys):
            #the gradient for each layer
            grads = sqerrorGradients(network,x,y)
           
            # Take a gradient step for each neuron in each layer
            network = [
                [gradient_step(neuron, grad, -lr)
            for neuron, grad in zip(layer, layer_grad)]
                
            for layer, layer_grad in zip(network, grads)
            ]
    print(feedForward(network, [1, 0])[-1][0])
    
    
if __name__ == "__main__": main()