from vectors import Vector, dot

#function to determine if a perceptron should fire
def stepFunction(x : float) -> float:
    return 1 if x>=0 else 0

#computes the weighted sum of inputs x
def perceptronOutput(weights: Vector, bias: float, x:Vector) -> float:
    #returns 1 if the perceptron should fire 0 if it shouldnt
    return stepFunction(dot(x,weights)+bias)


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
    print("done")
if __name__ == "__main__": main()