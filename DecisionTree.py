from typing import List,Any,NamedTuple, Optional,Dict, TypeVar
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
