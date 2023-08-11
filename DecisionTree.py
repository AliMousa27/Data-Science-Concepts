from typing import List,Any,NamedTuple, Optional,Dict, TypeVar,Union
import math
from collections import Counter,defaultdict

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None # allow unlabeled data
    
inputs = [#   level           lang         tweets        phd        did_well
    Candidate('Senior', 'Java', False, False, False),
    Candidate('Senior', 'Java', False, True, False),
    Candidate('Mid', 'Python', False, False, True),
    Candidate('Junior', 'Python', False, False, True),
    Candidate('Junior', 'R', True, False, True),
    Candidate('Junior', 'R', True, True, False),
    Candidate('Mid', 'R', True, True, True),
    Candidate('Senior', 'Python', False, False, False),
    Candidate('Senior', 'R', True, False, True),
    Candidate('Junior', 'Python', True, False, True),
    Candidate('Senior', 'Python', True, True, True),
    Candidate('Mid', 'Python', False, True, True),
    Candidate('Mid', 'Java', True, False, True),
    Candidate('Junior', 'Python', False, True, False)
    ]


def entropy(classProbabilities: List[float]) -> float:
    '''classProbabilities is a list of p values where each p value is
        the proportion of hoiw much of our data belongs to that class
    '''
    #if statement to ignore data with no classes
    return sum([-p*math.log(p,2) for p in classProbabilities if p > 0])


#our data is a list of (input,label) here we compute the percent of each label
def classProbabilities(labels: List[Any]) -> List[float]:
    n= len(labels)
    #return a list of proprotons of each label
    return [p/n for p in Counter(labels).values()]
#returns the entropy of a given data set
def dataEntropy(labels: List[Any]) -> float:
    return entropy(classProbabilities(labels))

def partitionEntropy(subsets: List[List[Any]]) -> float:
    totalCount = sum(len(subset) for subset in subsets)
    #the multipication and division is the fraction of how muhc data. that is the weight in this case
    # since we are trying to find a weighted sum
    return sum([dataEntropy(subset) * len(subset)/totalCount 
                for subset in subsets])
    


T = TypeVar('T') # generic type for inputs

def partitionByAttribute(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute."""
    #create a default dict, key is the attreiobvute val is a list of data that has said attribute
    partitions : Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        #get the key for each attribute (attributes are diffrent but all under same category)
        #example experince level is mid senior and junior
        key = getattr(input,attribute)
        #add the input to the list
        partitions[key].append(input)
    return partitions

def partitionEntropyByLabel(inputs: List[Any],
    attribute: str,
    label_attribute: str) -> float:
    """Compute the entropy corresponding to the given partition"""
    
    # partition based on the chosen attribute
    partitions = partitionByAttribute(inputs, attribute)
    #get the label we want for each attribute in the dict that contains the lists
    #e.x 'Senior': [Candidate(level='Senior', lang='Java', tweets=False, phd=False, did_well=False),.......
    # and 'Mid': [Candidate(level='Mid', lang='Python', tweets=False, phd=False, did_well=True)......
    # these are 2 lists where the key is the attribute type
    # then we get each label, for each of these lists, making it a 2d list
    labels = [[getattr(input, label_attribute) for input in partition]
                for partition in partitions.values()]
    return partitionEntropy(labels)
    
    
#choose the lowest entropy to partition by
minEntropyAttribute = min([partitionEntropyByLabel(inputs, key, 'did_well'),key]
                          for key in ['level','lang','tweets','phd'])

print(minEntropyAttribute)
#split based on lowest entropy which was level

seniorInputs = [input for input in inputs if input.level == 'Senior']
minEntropyAttribute = min([partitionEntropyByLabel(seniorInputs, key, 'did_well'),key]
                          for key in ['level','lang','tweets','phd'])
print(minEntropyAttribute)
#now we create a tree which is eaither a leaf with a value or a split 
class Leaf(NamedTuple):
    value: Any
    
'''a Split (containing an attribute to split on, subtrees for specific
values of that attribute, and possibly a default value to use if we
see an unknown value).
'''
class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None
    
DecisionTree = Union[Leaf, Split]

def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""
    #if the tree is down to a leaf node, just return its value
    if isinstance(tree,Leaf):
        return tree.value
    #otherwise we have a subtree we need to work through
    #subtree has a key to split on and a dict whose are to consider next
    #here we get the value of the next key to consider
    subtreeKey = getattr(input,tree.attribute)
    
    if subtreeKey not in tree.subtrees: # If no subtree for key,
        return tree.default_value # return the default value.
    
    subtree = tree.subtrees[subtreeKey] # Choose the appropriate subtree
    return classify(subtree, input) # and use it to classify the input.



def buildTreeID3(inputs: List[Any],
    splitAttributes: List[str],#attribute to split with
    targetAttribute: str) -> DecisionTree:
    
    # Count target labels
    labelCounts = Counter(getattr(input, targetAttribute)
    for input in inputs)
    
    mostCommonLabel = labelCounts.most_common(1)[0][0]
    
    # If there's a unique label, predict it
    if len(labelCounts) == 1:
        return Leaf(value=mostCommonLabel)
    # If no split attributes left, return the majority label
    if not splitAttributes:
        return Leaf(value=mostCommonLabel)
    
    # Otherwise split by the best attribute
    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partitionEntropyByLabel(inputs, attribute, targetAttribute)
    
    bestAttribute = min(splitAttributes, key=split_entropy)#make a list of each entropy with the list attribute, then get the minumum attribute of entropy
    partitions = partitionByAttribute(inputs, bestAttribute)#partition with said attribute
    newleftAttributes = [a for a in splitAttributes if a != bestAttribute]#get the rest of the attributes
    
    # Recursively build the subtrees
    subtrees = {attribute_value : buildTreeID3(inputs=subset,
    splitAttributes=newleftAttributes,
    targetAttribute=targetAttribute)
    for attribute_value, subset in partitions.items()}
    
    return Split(bestAttribute, subtrees, default_value=mostCommonLabel)

tree = buildTreeID3(inputs=inputs,
splitAttributes=['level', 'lang', 'tweets', 'phd'],
targetAttribute='did_well')
