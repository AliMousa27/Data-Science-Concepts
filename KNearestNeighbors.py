from typing import List,NamedTuple
from collections import Counter
data=["a","b","b","c","a"]
def majorityVote(labels: List[str]):
    """Assumes that labels are ordered from nearest to farthest."""
    voteCounts = Counter(labels)
    winner, winnerCount = voteCounts.most_common(1)[0]
    #find if we have the same amount of count for more than 1 item
    numOfWinners= len([val for val in  voteCounts.values() if val==winnerCount])
    if numOfWinners==1:
        return winner
    else:
        #try again without the farthest aka the last element to find a winner that ios truely closest to K
        return majorityVote(labels[:-1])

from vectors import distance,Vector
class LabeledPoint(NamedTuple):
    label:str
    point:Vector
    
def knnClassify(k:int,
                points:list[LabeledPoint],
                newPoint: LabeledPoint):
    #Order points from nearest to farthest
    sortedPoints = sorted(points,
                          #create a lambda function that gives the key to be the distance between the points to sort with
                          key=lambda point:distance(point,newPoint))
    #find labels for closest K neighbors(the [:k] means from start until k that is to say find k neaighest neighbords :) )
    labeledKNeighbors = [point.label for point in sortedPoints[:k]] 
    #then find the label that is most common to the new point
    return majorityVote(labeledKNeighbors)
