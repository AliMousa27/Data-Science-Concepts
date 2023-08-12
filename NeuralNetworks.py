from vectors import Vector, dot

#function to determine if a perceptron should fire
def stepFunction(x : float) -> float:
    return 1 if x>=0 else 0

#computes the weighted sum of inputs x
def perceptronOutput(weights: Vector, bias: float, x:Vector) -> float:
    #returns 1 if the perceptron should fire 0 if it shouldnt
    return stepFunction(dot(x,weights)+bias)

and_weights = [2., 2]
and_bias = -3.
assert perceptronOutput(and_weights, and_bias, [1, 1]) == 1
assert perceptronOutput(and_weights, and_bias, [0, 1]) == 0
assert perceptronOutput(and_weights, and_bias, [1, 0]) == 0
assert perceptronOutput(and_weights, and_bias, [0, 0]) == 0