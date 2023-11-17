from os import name
import random

import tqdm
from vectors import Vector,vector_mean,squared_distance
from typing import List
import itertools

#compute the number of differnece between 2 vectors
def numberOfDifferences(v1: Vector, v2: Vector) -> Vector:
    assert len(v1)==len(v2)
    return len([x1 for x1,x2 in zip(v1,v2) if x1!=x2])

#given a vector and their assigned clusters compute mean of cluster
def clusterMeans(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    #create k clusters
    clusters = [[] for _ in range(k)]
    for input,assignment in zip(inputs,assignments):
        # clusters[i] contains the inputs whose assignment is i
        #add the input to its assigned cluster
        clusters[assignment].append(input)
        #chose random input for empty cluster to be the mean
        return [vector_mean(cluster) if cluster else random.choice(inputs)
                for cluster in clusters]

        
class KMeans():
    def __init__(self, k:int) -> None:
        self.means = None
        self.k=k
        
    #returns the assigned the cluster to each input
    def classify(self,input: Vector)->int:
        #make an iterable of k-1 vals then squared distance between the input and each cluster
        #and return the index of the min cluster
        return min(range(self.k),
                   key=lambda i: squared_distance(self.means[i], input))
        
    def train(self,inputs:List[Vector]):
        #start with random assignment for each input vector
        assignments = [random.randrange(self.k) for _ in inputs]
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                #compute means
                self.means = clusterMeans(self.k,inputs,assignments)
                #find new assignments
                newAssignments = [self.classify(input) for input in inputs]
                numOfChanges = numberOfDifferences(assignments,newAssignments)
                if numOfChanges == 0:
                    return
                else:
                    self.means = clusterMeans(self.k,inputs,newAssignments)
                    assignments=newAssignments
                    
def main():
        
    inputs: List[List[float]] = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
    random.seed(12)                   
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)   
        
    assert len(means) == 3
        
    # Check that the means are close to what we expect.
    assert squared_distance(means[0], [-44, 5]) < 1
    assert squared_distance(means[1], [-16, -10]) < 1
    assert squared_distance(means[2], [18, 20]) < 1
        
    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
        
    assert len(means) == 2
    assert squared_distance(means[0], [-26, -5]) < 1
    assert squared_distance(means[1], [18, 20]) < 1
        
if name == "__main__":main()