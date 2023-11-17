from typing import List,Callable,Iterable,Tuple
from vectors import dot
from NeuralNetworks import argmax, sigmoid
from probabilty import inverse_normal_cdf
import random
import tqdm
import matplotlib.pyplot as plt
import mnist
import math
Tensor = list


def shape(tensor: Tensor)-> List[int]:
    """finds rows and columns of a given tensor"""
    sizes=[]
    while isinstance(tensor,list):
        sizes.append(len(tensor))
        tensor=tensor[0]
    return sizes

def is1D(tensor: Tensor) -> bool:
    """
    If tensor[0] is a list, it's a higher-order tensor.
    Otherwise, tensor is 1-dimensonal (that is, a vector).
    """
    return not isinstance(tensor[0],list)

def tensorSum(tensor: Tensor) -> float:
    """Sums up all the values in the tensor"""
    if is1D(tensor):
        return sum(tensor)#if its just a list just sum it
    else:
        return sum(tensorSum(t) for t in tensor)#call tensor sum on each row

def tensorApply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """Applies f elementwise"""
    if is1D(tensor):
        return [f(element) for element in tensor]
    else:
        return [tensorApply(f,t) for t in tensor]
    
def zeroesTensor(tensor : Tensor):
    """return a tensor filled with 0s with the same shape as the tensor given"""
    return tensorApply(lambda _: 0.0,tensor)

def tensorCombine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """Applies f to corresponding elements of t1 and t2"""
    if is1D(t1):
        return [f(x,y) for x, y in zip(t1,t2)]
    else:
        return [tensorCombine(f,t1_i,t2_i) for t1_i, t2_i in zip(t1,t2)]
    
class Layer:
    """
    Our neural networks will be composed of Layers, each of which
    knows how to do some computation on its inputs in the "forward"
    direction and propagate gradients in the "backward" direction.
    """
    def forward(self, input):
        """
        Note the lack of types. We're not going to be prescriptive
        about what kinds of inputs layers can take and what kinds
        of outputs they can return.
        """
        raise NotImplementedError

    def backward(self, gradient):
        """
        Similarly, we're not going to be prescriptive about what the
        gradient looks like. It's up to you the user to make sure
        that you're doing things sensibly.
        """
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """ note: params are basically the weights
        Returns the parameters of this layer. The default implementation
        returns nothing, so that if you have a layer with no parameters
        you don't have to implement this.
        """
        return ()

    def grads(self) -> Iterable[Tensor]:
        """
        Returns the gradients, in the same order as params()
        """
        return ()
    
class Sigmoid(Layer):
    def forward(self, input) -> Tensor:
        """apply sigmoid function to each element of the input tensor"""
        self.sigmoids = tensorApply(sigmoid,input)
        return self.sigmoids
    
    def backward(self, gradient):
        """apply each gradient to each element from the self.sigmoids"""
        #create a lambda function for the gradient of sigmoid with respect to the gradient
        f= lambda sig,grad: sig * (1-sig) * grad
        #then apply the derivative function to them and return the new tensor
        return tensorCombine(f,self.sigmoids,gradient)
    
def randomUniform(*dims)->Tensor:
    #dims is the shape of the tensor we want basically
    if len(dims)==1:
        #if we reach the lowest dimension then we create a list with the amount the dimension is
        return [random.random() for _ in range(dims[0])]
    else:
        #recurisvly go through the dimensions. 
        #it works by creating a list, popping the call stack then going again and adding
        #as many elemnts as the previos dim requires
        #e.x (2,4) we add 2 lists of 4 elements.
        #or (2,3,4) we create lists of 4 elemnts. 3 of these lists in the 2d lists and 2 of the 2d list 
        #to make a 3d list
        return [randomUniform(*dims[1:]) for _ in range(dims[0])] 
    
def randomNormal(*dims: int,
                  mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random())
                for _ in range(dims[0])]
    else:
        return [randomNormal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]
        
def randomTensor(*dims: int, init: str = 'normal') -> Tensor:
    """generate a random tensor based on the init parameter"""
    if init == 'normal':
        return randomNormal(*dims)
    elif init == 'uniform':
        return randomUniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return randomNormal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")
    
class Linear(Layer):
    #inputDim tells us how many weights a neuron needs to have same length vector for dot to work
    #output dim tells us how many neurons we need
    def __init__(self, inputDim,outputDim,init="xavier") -> None:
        """a layer of output dim neurons each with input dim weights and a bias"""
        self.inputDim=inputDim
        self.outputDim=outputDim
        
        #self.w[n] is the weights of the nth neuron 
        self.w = randomTensor(outputDim,inputDim, init=init)
        #add a bias to each out put neuron
        self.b = randomTensor(outputDim,init=init)
        
    def forward(self, input):
        #save input to use in backpropagation
        self.input = input
        #return one output per output neuron whcih is w * input + bias
        return [dot(self.w[n], input) + self.b[n]
                for n in range(self.outputDim)
                ]
        
    def backward(self, gradients: Tensor) -> Tensor:
    #gradients param is more or less the gradient of loss. so each derivative is with respec to it
    # Each b[o] gets added to output[o], which means
    # the gradient of b with respect to the output 
    # is the same as the output gradient.
        self.bGradients = gradients
        
        # Each w[o][i] multiplies input[i] and gets added to output[o].
        # So its gradient is input[i] * gradient[o].
        #all this to say we multiply each input by the oth gradient 
        self.wGradients = [[self.input[i] * gradients[o] for i in range(self.inputDim)] #iterate over each 
                           #input mutliplied by the gradient of the oth neuron
                           for o in range(self.outputDim)#iterate over each neuron
                           ]
        
        # Each input[i] multiplies every w[o][i] and gets added to every
        # output[o]. So its gradient is the sum of w[o][i] * gradient[o]
        # across all the outputs.
        return [sum(self.w[o][i] * gradients[o] for o in range(self.outputDim))
                for i in range(self.inputDim)]
        
        
    def params(self) -> Iterable[Tensor]:
        return [self.w,self.b]
    
    def grads(self) -> Iterable[Tensor]:
        return [self.wGradients,self.bGradients]
    
#this is a list of layers to represent a network
class Sequential(Layer):
    def __init__(self,layers: List[Layer]) -> None:
        self.layers=layers
        
    def forward(self,input):
        #just get the input of each layer in order
        for layer in self.layers:
            #make the input of the next layer equal to the output of the current layer we are on
            input = layer.forward(input)
        return input
    def backward(self, gradient):
        #backpropgate the gradinet for each layer in reverse
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def params(self) -> Iterable[Tensor]:
        #return the paramaters for each layer
        return (param for layer in self.layers
                for param in layer.params())
        
    def grads(self) -> Iterable[Tensor]:
        return (grad for layer in self.layers
                for grad in layer.grads())
        

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """How good are our predictions? (Larger numbers are worse.)"""
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """How does the loss change as the predictions change?"""
        raise NotImplementedError
    
#class to calc the squared error and its gradient
class SSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        #compute tensor of the squared difference
        f = lambda predicted,actual: (predicted-actual)**2
        #apply tensor combine so the function executes on 2 inputes to get the squared error
        squaredErrors = tensorCombine(f,predicted,actual)
        #then just return the sum
        return sum(squaredErrors)
        
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        #apply the gradient of the squared errror and return a tensor of gradients
        f = lambda pred, act : 2*(pred-act)
        return tensorCombine(f,predicted,actual)
    
class Optimizer:
    def step(layer: Layer)->None:
        #an optimizer takes a graidnet step for each of the weights of the neurons in the layer to 
        #to optimize them using info knwon by either the optimizer or the layer or both
        raise NotImplementedError
    
class GradientDescent(Optimizer):
    def __init__(self, learningRate:float = 0.01) -> None:
        self.learningRate = learningRate
        
    def step(self,layer: Layer) -> None:
        for param,grads in zip(layer.params(),layer.grads()):
            #update the params using a gradient step
            f = lambda param, grads : param - grads*self.learningRate
            #the [:] is used to actually change the values inside the list and not just change the local variable
            param[:] = tensorCombine(f,param,grads)
        
#class thats similar to grad descent but doesnt overreact to the gradient
class Momentum(Optimizer):
    def __init__(self, learningRate:float, momentum : float = 0.9) -> None:
        self.learningRate = learningRate
        self.momentum : momentum
        self.updates : List[Tensor]=[] #running average
        
    def step(self, layer: Layer) -> None:
        
        #if we have no updates start with zeroes as the updates
        if not self.updates:
            updates = [zeroesTensor(grad) for grad in layer.grads()]
        for update,param,grad in zip(self.updates,layer.params(),layer.grads()):    
            #apply momentum
            update[:] = tensorCombine(
                lambda u,g: self.momentum * u + (1-self.momentum) * g,
                update,
                grad
            )
            #take gradient stepm and update the params accoridngly for the current params of the neuron
            param[:] = tensorCombine(
                lambda p,u : p -self.learningRate * u,
                param,
                update
            )
          
def tanh(x: float) -> float:
    if x < -100: return -1
    elif x > 100: return 1
    else:
        #this is e^-2x
        em2x = math.exp(-2 * x)
        return (1 - em2x) / (1 + em2x)
  
#function to make a probabilty disturbution of the output layer of a neural network
def softmax(tensor: Tensor) -> Tensor:
    if is1D(tensor):
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]
        summation = sum(exps)
        return [num / summation for num in exps]
    else:
        return [softmax(arr) for arr in tensor]
        
class SoftmaxCrossEntropy(Loss):
    """
    This is the negative-log-likelihood of the observed values, given the
    neural net model. So if we choose weights to minimize it, our model will
    be maximizing the likelihood of the observed data.
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Apply softmax to get probabilities
        probabilities = softmax(predicted)

        # This will be log p_i for the actual class i and 0 for the other
        # classes. We add a tiny amount to p to avoid taking log(0).
        likelihoods = tensorCombine(lambda p, act: math.log(p + 1e-30) * act,
                                     probabilities,
                                     actual)

        # And then we just sum up the negatives.
        return -tensorSum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)

        # Isn't this a pleasant equation?
        return tensorCombine(lambda p, actual: p - actual,
                              probabilities,
                              actual)

class Dropout(Layer):
    def __init__(self,probability:float) -> None:
        self.train = True
        self.probability = probability
    def forward(self, input: Tensor) -> Tensor:
        if self.train:
            #create a mask of 0s and 1s with the same shape as the input
            self.mask = tensorApply(lambda _: 0 if random.random() < self.probability else 1,input)
            #apply the mask to the input
            return tensorCombine(lambda x,m: x*m,input,self.mask)
        else:
            #if we are not training just return the input
            return input
    def backward(self, gradient: Tensor) -> Tensor:
        if self.train:
            #apply the mask to the gradient
            return tensorCombine(lambda x,m: x*m,gradient,self.mask)
        else:
            raise RuntimeError("don't call backward when not in train mode")
    
class Tanh(Layer):
    def forward(self, input: Tensor) -> Tensor:
        #save tanh output to use in backpropagation
        #and apply tanh to each element of the input tensor
        self.tanh = tensorApply(tanh,input)
        return self.tanh
    
    def backward(self, gradient):
        derivative = lambda tanh,grad: 1-tanh**2 * grad
        return tensorCombine(derivative,self.tanh,gradient)

def encode(number : int, numOfLabels:int =10):
    return [1 if i == number else 0 for i in range(numOfLabels)]

def loop(model: Layer,
            images: List[Tensor],
            labels: List[Tensor],
            loss: Loss,
            optimizer: Optimizer = None) -> None:
    correct = 0         # Track number of correct predictions.
    total_loss = 0.0    # Track total loss.
    
    with tqdm.trange(len(images)) as t:
        for i in t:
            predicted = model.forward(images[i])             # Predict.
            if argmax(predicted) == argmax(labels[i]):       # Check for
                correct += 1                                 # correctness.
                total_loss += loss.loss(predicted, labels[i])    # Compute loss.
                #print(f"correct is {correct} out of {i}")
                # If we're training, backpropagate gradient and update weights.
            if optimizer is not None:
                gradient = loss.gradient(predicted, labels[i])
                model.backward(gradient)
                optimizer.step(model)
    
                # And update our metrics in the progress bar.
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            t.set_description(f"mnist loss: {avg_loss:.3f} acc: {acc:.3f}")
    #print(f"correct is {correct} out of {len(images)}")
def main():
    
    # This will download the data, change this to where you want it.
    # (Yes, it's a 0-argument function, that's what the library expects.)
    # (Yes, I'm assigning a lambda to a variable, like I said never to do.)
    mnist.temporary_dir = lambda: '/tmp'
    
    # Each of these functions first downloads the data and returns a numpy array.
    # We call .tolist() because our "tensors" are just lists.
    train_images = mnist.train_images().tolist()
    train_labels = mnist.train_labels().tolist()
    
    assert shape(train_images) == [60000, 28, 28]
    assert shape(train_labels) == [60000]
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(10, 10)
    
    for i in range(10):
        for j in range(10):
            # Plot each image in black and white and hide the axes.
            ax[i][j].imshow(train_images[10 * i + j], cmap='Greys')
            ax[i][j].xaxis.set_visible(False)
            ax[i][j].yaxis.set_visible(False)
    
    plt.show()
    

    test_images = mnist.test_images().tolist()
    test_labels = mnist.test_labels().tolist()
    
    assert shape(test_images) == [10000, 28, 28]
    assert shape(test_labels) == [10000]
    
    
    # Recenter the images
    
    # Compute the average pixel value
    avg = tensorSum(train_images) / 60000 / 28 / 28
    
    # Recenter, rescale, and flatten
    train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                    for image in train_images]
    test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                   for image in test_images]

    
    # After centering, average pixel should be very close to 0
    
    
    # One-hot encode the test data
    train_labels = [encode(label) for label in train_labels]
    test_labels = [encode(label) for label in test_labels]
    
    #train_labels = [tensorSum(label) for label in train_labels]
    #test_labels = [tensorSum(label) for label in test_labels]
    
    
    
    # Training loop
    
    
    # The logistic regression model for MNIST

    
    random.seed(0)
    
    model = Linear(784, 10)
    loss = SoftmaxCrossEntropy()
    
    # This optimizer seems to work
    optimizer = GradientDescent(learningRate=0.01)
    
    # Train on the training data
    loop(model, train_images, train_labels, loss, optimizer)
    
    # Test on the test data (no optimizer means just evaluate)
    loop(model, test_images, test_labels, loss)
    

    
    # XOR
    # training data
    '''xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]
    
    network = Sequential(
        [
            Linear(2,2),#first layer takaes 2 and outputs 2
            Sigmoid(),#sigmoid turns the 2 inputs into 2 outputs
            Linear(2,1)#take 2 inputs and outputs 1 number
        ]
    )
    optimizer = GradientDescent()
    loss = SSE()
    
    with tqdm.trange(30000) as t:
        for epoch in t:
            for x,y in zip(xs,ys):
                predicted = network.forward(x)
                gradient =loss.gradient(predicted,y)
                network.backward(gradient)
                
                optimizer.step(network)'''

                
    

if __name__ =="__main__":main()