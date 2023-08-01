from typing import List,NamedTuple
from collections import Counter
import requests
import csv
from vectors import distance,Vector

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


def parseIrisRow(row: List[str])->LabeledPoint:
    #parse the list of strings where the first 4 values are a measurment vector
    #and the last fifth value is the label aka the species
    
    #get everything beside the species
    measurements = [float(val) for val in row[:-1]]
    #species is for example Iris-setosa but we just want the latter part
    label= row[-1].split("-")[-1]
    return LabeledPoint(label,measurements)

data: List[LabeledPoint] =[]

with open("Iris.dat","r") as f:
    csvReader=csv.reader(f,delimiter=",")
    for row in csvReader:
        data.append(parseIrisRow(row))
        
print(data)